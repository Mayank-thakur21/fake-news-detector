"""Fetch a news URL and extract the article title + body text.

Security: every hop (including redirects) is resolved and rejected if it
points at a private, loopback, link-local or otherwise non-public address,
preventing the API from being used for SSRF against internal services.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from backend.core.errors import ScrapeError, UpstreamError

logger = logging.getLogger(__name__)

MAX_REDIRECTS = 5
_STRIP_TAGS = (
    "script",
    "style",
    "noscript",
    "nav",
    "footer",
    "header",
    "aside",
    "form",
    "iframe",
    "svg",
    "figure",
    "button",
)
_MIN_PARAGRAPH_CHARS = 40


@dataclass(frozen=True)
class ScrapedArticle:
    url: str
    final_url: str
    title: str | None
    text: str
    site_name: str | None
    published_at: str | None
    word_count: int


class ArticleScraper:
    def __init__(
        self,
        *,
        timeout: float = 10.0,
        max_bytes: int = 5_000_000,
        user_agent: str = "FakeNewsDetector/2.0",
        allow_private_hosts: bool = False,
        session: requests.Session | None = None,
    ):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.allow_private_hosts = allow_private_hosts
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5",
                "Accept-Language": "en;q=0.9",
            }
        )

    # ------------------------------------------------------------------ #
    # URL safety
    # ------------------------------------------------------------------ #
    def validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ScrapeError("Only http(s) URLs are supported")
        if not parsed.hostname:
            raise ScrapeError("URL has no hostname")
        if self.allow_private_hosts:
            return
        try:
            infos = socket.getaddrinfo(
                parsed.hostname, parsed.port or None, proto=socket.IPPROTO_TCP
            )
        except socket.gaierror as exc:
            raise ScrapeError(f"Could not resolve host '{parsed.hostname}'") from exc
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global or ip.is_multicast:
                raise ScrapeError(f"Refusing to fetch non-public address ({parsed.hostname})")

    # ------------------------------------------------------------------ #
    # Fetching
    # ------------------------------------------------------------------ #
    def fetch_html(self, url: str) -> tuple[str, str]:
        """Return ``(final_url, html)``, following redirects manually and safely."""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            self.validate_url(current)
            try:
                resp = self.session.get(
                    current, timeout=self.timeout, allow_redirects=False, stream=True
                )
            except requests.Timeout as exc:
                raise UpstreamError(f"Timed out fetching {current}") from exc
            except requests.RequestException as exc:
                raise UpstreamError(f"Could not fetch URL: {exc.__class__.__name__}") from exc

            try:
                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        raise UpstreamError("Redirect without Location header")
                    current = urljoin(current, location)
                    continue
                if resp.status_code >= 500:
                    raise UpstreamError(f"Remote server returned HTTP {resp.status_code}")
                if resp.status_code >= 400:
                    raise ScrapeError(f"Remote server returned HTTP {resp.status_code}")
                ctype = resp.headers.get("Content-Type", "").lower()
                if ctype and "html" not in ctype and "xml" not in ctype:
                    raise ScrapeError(f"URL does not point to an HTML page ({ctype.split(';')[0]})")
                return current, self._read_limited(resp)
            finally:
                resp.close()
        raise ScrapeError(f"Too many redirects (>{MAX_REDIRECTS})")

    def _read_limited(self, resp: requests.Response) -> str:
        chunks, size = [], 0
        for chunk in resp.iter_content(chunk_size=65536):
            size += len(chunk)
            if size > self.max_bytes:
                raise ScrapeError(f"Page is larger than {self.max_bytes // 1_000_000} MB")
            chunks.append(chunk)
        raw = b"".join(chunks)
        encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        return raw.decode(encoding, errors="replace")

    # ------------------------------------------------------------------ #
    # Extraction
    # ------------------------------------------------------------------ #
    @staticmethod
    def extract(html: str) -> tuple[str | None, str, str | None, str | None]:
        """Return ``(title, text, site_name, published_at)`` from raw HTML."""
        soup = BeautifulSoup(html, "lxml")

        def meta(*names: str) -> str | None:
            for n in names:
                tag = soup.find("meta", attrs={"property": n}) or soup.find(
                    "meta", attrs={"name": n}
                )
                if tag and tag.get("content"):
                    return tag["content"].strip()
            return None

        title = meta("og:title", "twitter:title")
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()
        site_name = meta("og:site_name", "application-name")
        published = meta("article:published_time", "datePublished", "pubdate", "date")

        for tag in soup(_STRIP_TAGS):
            tag.decompose()

        # Prefer semantic containers, fall back to the whole document.
        container = (
            soup.find("article")
            or soup.find(attrs={"itemprop": "articleBody"})
            or soup.find("main")
            or soup.body
            or soup
        )
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in container.find_all("p")
            if len(p.get_text(strip=True)) >= _MIN_PARAGRAPH_CHARS
        ]
        if not paragraphs and container is not soup:  # article tag with no <p>s
            paragraphs = [
                p.get_text(" ", strip=True)
                for p in soup.find_all("p")
                if len(p.get_text(strip=True)) >= _MIN_PARAGRAPH_CHARS
            ]
        text = "\n\n".join(paragraphs)
        if not text:
            text = meta("og:description", "description") or ""
        return title, text, site_name, published

    def scrape(self, url: str) -> ScrapedArticle:
        logger.info("Scraping %s", url)
        final_url, html = self.fetch_html(url)
        title, text, site_name, published = self.extract(html)
        if len(text.split()) < 20:
            raise ScrapeError(
                "Could not extract enough article text from this page "
                "(it may be paywalled or rendered with JavaScript). Paste the text instead."
            )
        return ScrapedArticle(
            url=url,
            final_url=final_url,
            title=title,
            text=text,
            site_name=site_name,
            published_at=published,
            word_count=len(text.split()),
        )

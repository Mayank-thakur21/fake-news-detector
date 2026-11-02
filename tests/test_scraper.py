from __future__ import annotations

import socket

import pytest
import requests

from backend.core.errors import ScrapeError, UpstreamError
from backend.services.scraper import ArticleScraper

ARTICLE_HTML = """
<html><head>
  <title>Fallback title</title>
  <meta property="og:title" content="Council approves transit budget">
  <meta property="og:site_name" content="Example Times">
  <meta property="article:published_time" content="2026-01-02T10:00:00Z">
</head><body>
  <nav><p>Home | World | Politics | Sport | Weather | Subscribe to our newsletter today</p></nav>
  <article>
    <p>The city council voted on Tuesday to approve a revised transit budget for next year.</p>
    <p>Officials said the plan will be funded by a combination of state grants and fare revenue.</p>
    <p>Short.</p>
  </article>
  <script>var tracking = "should never appear in the extracted article text at all";</script>
  <footer><p>Copyright Example Times. All rights reserved. Terms and privacy apply.</p></footer>
</body></html>
"""


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.encoding = "utf-8"
        self.apparent_encoding = "utf-8"

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308) and "Location" in self.headers

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self):
        pass


class FakeSession(requests.Session):
    def __init__(self, responses):
        super().__init__()
        self.responses = dict(responses)
        self.calls: list[str] = []

    def get(self, url, **kwargs):  # type: ignore[override]
        self.calls.append(url)
        resp = self.responses[url]
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def public_dns(monkeypatch):
    """Resolve every hostname to a public IP except *.internal -> 10.0.0.5."""

    def fake_getaddrinfo(host, *args, **kwargs):
        ip = "10.0.0.5" if host.endswith(".internal") else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


def make(responses, **kw) -> ArticleScraper:
    return ArticleScraper(session=FakeSession(responses), **kw)


def test_extract_prefers_article_and_metadata():
    title, text, site, published = ArticleScraper.extract(ARTICLE_HTML)
    assert title == "Council approves transit budget"
    assert site == "Example Times"
    assert published == "2026-01-02T10:00:00Z"
    assert "transit budget" in text and "state grants" in text
    assert "tracking" not in text and "Copyright" not in text and "Subscribe" not in text
    assert "Short." not in text


def test_extract_falls_back_to_meta_description():
    html = '<html><head><meta name="description" content="Only a description"></head></html>'
    _, text, _, _ = ArticleScraper.extract(html)
    assert text == "Only a description"


def test_scrape_success(public_dns):
    url = "https://news.example.com/story"
    s = make({url: FakeResponse(body=ARTICLE_HTML.encode())})
    art = s.scrape(url)
    assert art.final_url == url
    assert art.word_count > 20


def test_scrape_follows_redirects(public_dns):
    a, b = "https://example.com/a", "https://example.com/b"
    s = make(
        {
            a: FakeResponse(301, headers={"Location": "/b"}),
            b: FakeResponse(body=ARTICLE_HTML.encode()),
        }
    )
    assert s.scrape(a).final_url == b


def test_redirect_to_private_host_is_blocked(public_dns):
    a = "https://example.com/a"
    s = make({a: FakeResponse(302, headers={"Location": "http://db.internal/admin"})})
    with pytest.raises(ScrapeError, match="non-public"):
        s.scrape(a)


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/", "http://localhost/", "http://169.254.169.254/latest/meta-data"]
)
def test_private_addresses_rejected(url):
    with pytest.raises(ScrapeError):
        ArticleScraper().validate_url(url)


def test_allow_private_hosts_flag():
    ArticleScraper(allow_private_hosts=True).validate_url("http://127.0.0.1:8000/x")


@pytest.mark.parametrize("url", ["ftp://example.com/file", "file:///etc/passwd", "http:///nohost"])
def test_bad_schemes_rejected(url):
    with pytest.raises(ScrapeError):
        ArticleScraper().validate_url(url)


def test_too_many_redirects(public_dns):
    url = "https://example.com/loop"
    s = make({url: FakeResponse(302, headers={"Location": url})})
    with pytest.raises(ScrapeError, match="Too many redirects"):
        s.scrape(url)


def test_non_html_rejected(public_dns):
    url = "https://example.com/file.pdf"
    s = make({url: FakeResponse(headers={"Content-Type": "application/pdf"})})
    with pytest.raises(ScrapeError, match="HTML"):
        s.scrape(url)


def test_size_limit(public_dns):
    url = "https://example.com/huge"
    s = make({url: FakeResponse(body=b"x" * 20_000)}, max_bytes=10_000)
    with pytest.raises(ScrapeError, match="larger than"):
        s.scrape(url)


def test_http_errors_map_to_exceptions(public_dns):
    u404, u503 = "https://example.com/404", "https://example.com/503"
    s = make({u404: FakeResponse(404), u503: FakeResponse(503)})
    with pytest.raises(ScrapeError) as e404:
        s.scrape(u404)
    assert not isinstance(e404.value, UpstreamError)
    with pytest.raises(UpstreamError):
        s.scrape(u503)


def test_timeout_is_upstream_error(public_dns):
    url = "https://example.com/slow"
    s = make({url: requests.Timeout("slow")})
    with pytest.raises(UpstreamError, match="Timed out"):
        s.scrape(url)


def test_page_without_article_text(public_dns):
    url = "https://example.com/empty"
    s = make({url: FakeResponse(body=b"<html><body><div>JS app</div></body></html>")})
    with pytest.raises(ScrapeError, match="enough article text"):
        s.scrape(url)

"""Per-session light/dark theming.

Streamlit's built-in theme is process-wide (changing it would flip the theme
for *every* connected user), so we inject scoped CSS instead and keep the
choice in ``st.session_state``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    muted: str
    accent: str
    fake: str
    real: str
    on_accent: str  # text colour on accent-coloured backgrounds
    plotly_template: str


LIGHT = Palette(
    name="light",
    bg="#f7f8fa",
    surface="#ffffff",
    surface_alt="#eef1f5",
    border="#d9dee5",
    text="#1b1f24",
    muted="#5b6573",
    accent="#2563eb",
    fake="#dc2626",
    real="#16a34a",
    on_accent="#ffffff",
    plotly_template="plotly_white",
)

DARK = Palette(
    name="dark",
    bg="#0e1117",
    surface="#161b22",
    surface_alt="#1f2630",
    border="#30363d",
    text="#e6edf3",
    muted="#9aa4b2",
    accent="#60a5fa",
    fake="#f87171",
    real="#4ade80",
    on_accent="#0b1220",
    plotly_template="plotly_dark",
)


def palette(mode: str) -> Palette:
    return DARK if mode == "dark" else LIGHT


def css(p: Palette) -> str:
    """CSS overriding Streamlit's base theme with the selected palette."""
    return f"""
<style>
:root {{
  --fnd-bg: {p.bg}; --fnd-surface: {p.surface}; --fnd-surface-alt: {p.surface_alt};
  --fnd-border: {p.border}; --fnd-text: {p.text}; --fnd-muted: {p.muted};
  --fnd-accent: {p.accent}; --fnd-on-accent: {p.on_accent};
  --fnd-fake: {p.fake}; --fnd-real: {p.real};
}}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
  background: var(--fnd-bg); color: var(--fnd-text);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{
  background: var(--fnd-surface); border-right: 1px solid var(--fnd-border);
}}
.stApp p, .stApp li, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6, .stApp td, .stApp th, .stApp small,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"],
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: var(--fnd-text); }}
[data-testid="stCaptionContainer"], .stApp .fnd-muted {{ color: var(--fnd-muted) !important; }}
textarea, input, [data-baseweb="select"] > div, [data-baseweb="input"] > div,
[data-baseweb="textarea"], [data-baseweb="base-input"] {{
  background: var(--fnd-surface) !important; color: var(--fnd-text) !important;
  border-color: var(--fnd-border) !important;
}}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background: var(--fnd-surface) !important; }}
[data-baseweb="popover"] li {{ color: var(--fnd-text) !important; }}
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"] details,
[data-testid="stMetric"], [data-testid="stFileUploaderDropzone"] {{
  background: var(--fnd-surface); border-color: var(--fnd-border) !important;
}}
[data-testid="stMetric"] {{
  border: 1px solid var(--fnd-border); border-radius: 10px; padding: 12px 16px;
}}
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 8px; border: 1px solid var(--fnd-border);
  background: var(--fnd-surface-alt); color: var(--fnd-text);
}}
.stButton > button[kind="primary"], [data-testid="stBaseButton-primary"] {{
  background: var(--fnd-accent) !important; border-color: var(--fnd-accent) !important;
  color: var(--fnd-on-accent) !important;
}}
.stButton > button[kind="primary"] p, [data-testid="stBaseButton-primary"] p {{
  color: var(--fnd-on-accent) !important;
}}
[data-baseweb="tab-list"] button p {{ color: var(--fnd-muted); }}
[data-baseweb="tab-list"] button[aria-selected="true"] p {{ color: var(--fnd-accent); }}
code {{ background: var(--fnd-surface-alt) !important; color: var(--fnd-text) !important; }}

/* --- custom components --- */
.fnd-verdict {{
  border-radius: 12px; padding: 18px 22px; margin: 6px 0 14px 0;
  border: 1px solid var(--fnd-border); background: var(--fnd-surface);
}}
.fnd-verdict.fake {{ border-left: 6px solid var(--fnd-fake); }}
.fnd-verdict.real {{ border-left: 6px solid var(--fnd-real); }}
.fnd-verdict .label {{ font-size: 1.6rem; font-weight: 700; }}
.fnd-verdict.fake .label {{ color: var(--fnd-fake); }}
.fnd-verdict.real .label {{ color: var(--fnd-real); }}
.fnd-verdict .meta {{ color: var(--fnd-muted); font-size: .9rem; margin-top: 4px; }}
.fnd-table-wrap {{ max-height: 460px; overflow: auto; border: 1px solid var(--fnd-border);
  border-radius: 10px; }}
table.fnd-table {{ width: 100%; border-collapse: collapse; font-size: .9rem; }}
table.fnd-table th {{ position: sticky; top: 0; background: var(--fnd-surface-alt);
  text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--fnd-border); }}
table.fnd-table td {{ padding: 7px 10px; border-bottom: 1px solid var(--fnd-border);
  background: var(--fnd-surface); vertical-align: top; }}
.fnd-pill {{ display: inline-block; padding: 1px 10px; border-radius: 999px; font-weight: 600;
  font-size: .8rem; }}
.fnd-pill.fake {{
  background: color-mix(in srgb, var(--fnd-fake) 18%, transparent); color: var(--fnd-fake);
}}
.fnd-pill.real {{
  background: color-mix(in srgb, var(--fnd-real) 18%, transparent); color: var(--fnd-real);
}}
</style>
"""

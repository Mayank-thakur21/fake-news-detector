"""Reusable rendering helpers for the Streamlit UI."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import Palette


def verdict_card(result: dict[str, Any]) -> None:
    is_fake = result["prediction"] == "Fake News"
    cls = "fake" if is_fake else "real"
    icon = "❌" if is_fake else "✅"
    meta = (
        f"{html.escape(result.get('model_name', ''))} · {result['confidence']:.1f}% confidence"
        f" · {result.get('processing_ms', 0):.0f} ms"
    )
    st.markdown(
        f'<div class="fnd-verdict {cls}"><div class="label">{icon} {result["prediction"]}</div>'
        f'<div class="meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )


def confidence_gauge(result: dict[str, Any], p: Palette) -> go.Figure:
    is_fake = result["prediction"] == "Fake News"
    color = p.fake if is_fake else p.real
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=result["confidence"],
            number={"suffix": "%", "font": {"color": p.text}},
            title={"text": "Confidence", "font": {"color": p.muted, "size": 14}},
            gauge={
                "axis": {"range": [50, 100], "tickcolor": p.muted},
                "bar": {"color": color},
                "bgcolor": p.surface_alt,
                "borderwidth": 0,
                "steps": [
                    {"range": [50, 70], "color": p.surface},
                    {"range": [70, 90], "color": p.surface_alt},
                ],
            },
        )
    )
    fig = _style(fig, p, height=240)
    fig.update_layout(margin={"l": 30, "r": 30, "t": 50, "b": 10})
    return fig


def probability_bar(result: dict[str, Any], p: Palette) -> go.Figure:
    probs = result["probabilities"]
    fig = go.Figure(
        go.Bar(
            x=[probs["fake"] * 100, probs["real"] * 100],
            y=["Fake", "Real"],
            orientation="h",
            marker_color=[p.fake, p.real],
            text=[f"{probs['fake'] * 100:.1f}%", f"{probs['real'] * 100:.1f}%"],
            textposition="auto",
        )
    )
    fig.update_xaxes(range=[0, 100], title="Probability (%)")
    return _style(fig, p, height=180)


def feature_breakdown(explanation: dict[str, Any], p: Palette) -> go.Figure:
    """Diverging bar chart: left = evidence for fake, right = evidence for real."""
    rows = [
        (c["term"] + ("" if c["present"] else " (absent)"), c["weight"], c["share"])
        for c in explanation["toward_fake"] + explanation["toward_real"]
    ]
    rows.sort(key=lambda r: r[1])
    fig = go.Figure(
        go.Bar(
            x=[r[1] for r in rows],
            y=[r[0] for r in rows],
            orientation="h",
            marker_color=[p.fake if r[1] < 0 else p.real for r in rows],
            customdata=[r[2] for r in rows],
            hovertemplate="<b>%{y}</b><br>weight %{x:.4f}<br>%{customdata:.1f}% of evidence"
            "<extra></extra>",
        )
    )
    fig.update_xaxes(title="◀ pushes toward FAKE  |  pushes toward REAL ▶", zeroline=True)
    return _style(fig, p, height=max(260, 26 * len(rows) + 80))


def feature_table(explanation: dict[str, Any]) -> pd.DataFrame:
    data = [
        {
            "Direction": "Fake" if c["weight"] < 0 else "Real",
            "Term": c["term"],
            "In text": "yes" if c["present"] else "no",
            "Share of evidence (%)": c["share"],
            "Weight": c["weight"],
        }
        for c in explanation["toward_fake"] + explanation["toward_real"]
    ]
    return pd.DataFrame(data).sort_values("Share of evidence (%)", ascending=False)


def metrics_chart(models: list[dict[str, Any]], p: Palette) -> go.Figure:
    metric_names = ["accuracy", "precision", "recall", "f1"]
    colors = [p.accent, p.real, p.fake, p.muted]
    fig = go.Figure()
    for metric, color in zip(metric_names, colors, strict=True):
        fig.add_bar(
            name=metric.capitalize(),
            x=[m["name"] for m in models],
            y=[(m["metrics"].get(metric) or 0) * 100 for m in models],
            marker_color=color,
            hovertemplate="%{x}<br>" + metric + ": %{y:.2f}%<extra></extra>",
        )
    lows = [(m["metrics"].get("accuracy") or 0) * 100 for m in models]
    floor = max(0, min(lows) - 10) if lows else 0
    fig.update_layout(barmode="group", legend={"orientation": "h", "y": 1.12})
    fig.update_yaxes(range=[floor, 100], title="Score (%)")
    return _style(fig, p, height=360)


def compare_chart(results: list[dict[str, Any]], p: Palette) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(
        name="Fake",
        x=[r["model_name"] for r in results],
        y=[r["probabilities"]["fake"] * 100 for r in results],
        marker_color=p.fake,
    )
    fig.add_bar(
        name="Real",
        x=[r["model_name"] for r in results],
        y=[r["probabilities"]["real"] * 100 for r in results],
        marker_color=p.real,
    )
    fig.update_layout(barmode="stack", legend={"orientation": "h", "y": 1.12})
    fig.update_yaxes(range=[0, 100], title="Probability (%)")
    fig.add_hline(y=50, line_dash="dot", line_color=p.muted)
    return _style(fig, p, height=320)


def donut(labels: list[str], values: list[int], colors: list[str], p: Palette) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.55, marker_colors=colors))
    return _style(fig, p, height=260)


def html_table(df: pd.DataFrame, *, pill_column: str | None = None) -> None:
    """Theme-aware HTML table (st.dataframe renders on a canvas CSS can't reach)."""
    if df.empty:
        st.info("Nothing to show yet.")
        return
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = "" if pd.isna(row[col]) else str(row[col])
            if col == pill_column and val:
                cls = "fake" if "fake" in val.lower() else "real"
                cells.append(f'<td><span class="fnd-pill {cls}">{html.escape(val)}</span></td>')
            else:
                cells.append(f"<td>{html.escape(val)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        f'<div class="fnd-table-wrap"><table class="fnd-table"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(body)}</tbody></table></div>",
        unsafe_allow_html=True,
    )


def _style(fig: go.Figure, p: Palette, *, height: int) -> go.Figure:
    fig.update_layout(
        template=p.plotly_template,
        height=height,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": p.text},
    )
    return fig

"""Streamlit UI for the Fake News Detection system.

Run with::

    streamlit run frontend/streamlit_app.py
"""

from __future__ import annotations

import logging
import os
from datetime import datetime

import pandas as pd
import streamlit as st

import components as ui
from api_client import ApiClient, ApiError
from history import make_entry, parse_batch_csv, parse_batch_text, to_csv_bytes, to_dataframe
from theme import css, palette

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("frontend")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODEL_OPTIONS = {
    "Server default": None,
    "Random Forest": "rf",
    "Logistic Regression": "lr",
    "Linear SVM": "svm",
}

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
st.session_state.setdefault("history", [])
st.session_state.setdefault("theme", os.getenv("DEFAULT_THEME", "light"))
st.session_state.setdefault("api_key", os.getenv("API_KEY", ""))

# --------------------------------------------------------------------------- #
# Sidebar: navigation, theme, connection
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📰 Fake News Detector")
    page = st.radio(
        "Navigation",
        ["Home", "Analyze", "Batch", "Compare models", "History", "About"],
        label_visibility="collapsed",
    )
    st.divider()
    dark = st.toggle("🌙 Dark mode", value=st.session_state.theme == "dark")
    st.session_state.theme = "dark" if dark else "light"
    with st.expander("⚙️ Connection"):
        st.caption(f"Backend: `{API_URL}`")
        st.session_state.api_key = st.text_input(
            "API key",
            value=st.session_state.api_key,
            type="password",
            help="Required when the backend has API_KEYS configured.",
        )

P = palette(st.session_state.theme)
st.markdown(css(P), unsafe_allow_html=True)
client = ApiClient(API_URL, api_key=st.session_state.api_key or None)


def chart(fig) -> None:
    """Render a Plotly figure with *our* template (Streamlit's theme is light-only here)."""
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def call(fn, *args, **kwargs):
    """Run an API call, rendering a friendly error instead of a traceback."""
    try:
        return fn(*args, **kwargs)
    except ApiError as exc:
        logger.warning("API error (%s): %s", exc.status, exc.message)
        if exc.status == 401:
            st.error("🔑 " + exc.message + " - set your API key under **⚙️ Connection**.")
        elif exc.status == 429:
            st.warning("⏳ " + exc.message)
        else:
            st.error(exc.message)
        return None


def record(result: dict, preview: str, source: str) -> None:
    st.session_state.history.insert(0, make_entry(result, input_preview=preview, source=source))


def show_result(result: dict, *, article_text: str | None = None) -> None:
    left, right = st.columns([3, 2])
    with left:
        ui.verdict_card(result)
        src = result.get("source", {})
        if src.get("type") == "url":
            st.markdown(
                f"**{src.get('title') or 'Untitled article'}**  \n"
                f"{src.get('site_name') or ''} · {src.get('word_count', 0)} words · "
                f"[open source]({src.get('final_url') or src.get('url')})"
            )
        chart(ui.probability_bar(result, P))
    with right:
        chart(ui.confidence_gauge(result, P))

    exp = result.get("explanation")
    if exp and (exp["toward_fake"] or exp["toward_real"]):
        st.subheader("🔬 Confidence breakdown by feature")
        st.caption(
            f"Method: {exp['method']}. Each bar is a word/phrase from the model's vocabulary; "
            "its length is how strongly it moved the prediction."
            + (f" Prior P(real) = {exp['base_value']:.2f}." if exp.get("base_value") else "")
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            chart(ui.feature_breakdown(exp, P))
        with c2:
            ui.html_table(ui.feature_table(exp), pill_column="Direction")


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.title("🔍 AI Fake News Detection")
    st.markdown(
        "Classify news articles as **Fake** or **Real** with three machine-learning models, "
        "see *why* the model decided, analyse URLs or whole batches, and export your results."
    )
    health = call(client.health)
    if health:
        if health["status"] == "ok":
            st.success(f"✅ Backend online · models loaded: {', '.join(health['models_loaded'])}")
        else:
            st.warning(f"⚠️ Backend degraded: {health.get('detail')}")
    stats = call(client.stats) if health else None
    if stats:
        c = st.columns(4)
        c[0].metric("Total predictions", f"{stats['total_predictions']:,}")
        c[1].metric("Last 24 hours", f"{stats['last_24h']:,}")
        avg = stats.get("average_confidence")
        c[2].metric("Avg. confidence", f"{avg:.1f}%" if avg is not None else "-")
        best = stats["models"].get("best_model")
        best_acc = stats["model_accuracy"].get(best) if best else None
        c[3].metric(
            "Best model accuracy",
            f"{best_acc * 100:.2f}%" if best_acc else "-",
            help=f"Held-out test accuracy of {best}",
        )

        by_label = stats.get("by_label", {})
        if by_label:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### Predictions by label")
                labels = list(by_label)
                colors = [P.fake if "Fake" in lbl else P.real for lbl in labels]
                chart(ui.donut(labels, list(by_label.values()), colors, P))
            with col2:
                st.markdown("##### Model accuracy (held-out test set)")
                ui.html_table(
                    pd.DataFrame(
                        [
                            {
                                "Model": m["name"],
                                "Accuracy (%)": round((m["metrics"].get("accuracy") or 0) * 100, 2),
                                "F1 (%)": round((m["metrics"].get("f1") or 0) * 100, 2),
                            }
                            for m in stats["models"]["models"]
                            if m["loaded"]
                        ]
                    )
                )
        dataset = stats["models"].get("dataset") or {}
        if "sample" in str(dataset.get("name", "")):
            st.info(
                "ℹ️ The models were trained on the small **demo sample**. Train on the full "
                "Kaggle dataset for reliable predictions (see README)."
            )

    with st.container(border=True):
        st.markdown("### 📖 How to use")
        st.markdown(
            "1. **Analyze** – paste article text *or* a news URL.\n"
            "2. **Batch** – analyse many articles at once (paste or upload CSV).\n"
            "3. **Compare models** – see how Random Forest, Logistic Regression and SVM differ.\n"
            "4. **History** – review and export everything you analysed as CSV."
        )


def page_analyze() -> None:
    st.title("📰 Analyze an article")
    mode = st.radio("Input", ["Text", "URL"], horizontal=True)
    c1, c2, c3 = st.columns([2, 1, 1])
    model_label = c1.selectbox("Model", list(MODEL_OPTIONS))
    explain = c2.toggle("Explain prediction", value=True)
    top_k = c3.slider("Features per side", 3, 20, 8, disabled=not explain)

    text = url = None
    if mode == "Text":
        text = st.text_area(
            "Article text", height=260, placeholder="Paste the full news article here…"
        )
    else:
        url = st.text_input("Article URL", placeholder="https://www.example.com/news/article")

    if st.button("🔍 Analyze", type="primary", width="stretch"):
        if mode == "Text" and not (text or "").strip():
            st.warning("Please paste an article first.")
            return
        if mode == "URL" and not (url or "").strip():
            st.warning("Please enter a URL first.")
            return
        with st.spinner("Fetching and analysing…" if url else "Analysing…"):
            result = call(
                client.predict,
                text=text,
                url=url.strip() if url else None,
                model=MODEL_OPTIONS[model_label],
                explain=explain,
                top_k=top_k,
            )
        if result:
            record(result, url or text or "", "URL" if url else "Text")
            show_result(result)


def page_batch() -> None:
    st.title("📚 Batch analysis")
    st.caption("Analyse up to 25 articles in one request. One bad item won't fail the batch.")
    tab_paste, tab_csv = st.tabs(["✍️ Paste", "📄 Upload CSV"])
    items: list[dict[str, str]] = []
    with tab_paste:
        raw = st.text_area(
            "Articles",
            height=260,
            placeholder=(
                "First article text…\n---\nSecond article text…\n---\n"
                "https://example.com/news/story"
            ),
            help="Separate articles with a line containing only ---. "
            "A block that is just a URL is fetched.",
        )
        if raw.strip():
            items = parse_batch_text(raw)
    with tab_csv:
        upload = st.file_uploader("CSV with a `text` and/or `url` column", type=["csv"])
        if upload is not None:
            try:
                items = parse_batch_csv(pd.read_csv(upload))
            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not read CSV: {exc}")
    model_label = st.selectbox("Model", list(MODEL_OPTIONS), key="batch_model")
    st.caption(f"{len(items)} article(s) ready")

    if st.button("🚀 Run batch", type="primary", disabled=not items, width="stretch"):
        with st.spinner(f"Analysing {len(items)} article(s)…"):
            resp = call(client.predict_batch, items, model=MODEL_OPTIONS[model_label])
        if not resp:
            return
        s = resp["summary"]
        c = st.columns(4)
        c[0].metric("Analysed", f"{s['succeeded']}/{s['total']}")
        c[1].metric("Fake", s["fake"])
        c[2].metric("Real", s["real"])
        c[3].metric(
            "Avg. confidence", f"{s['average_confidence']:.1f}%" if s["average_confidence"] else "-"
        )
        rows = []
        for r, item in zip(resp["results"], items, strict=True):
            preview = (item.get("url") or item.get("text", ""))[:100]
            if r["ok"]:
                res = r["result"]
                record(res, item.get("url") or item.get("text", ""), "Batch")
                rows.append(
                    {
                        "#": r["index"] + 1,
                        "Input": preview,
                        "Prediction": res["prediction"],
                        "Confidence (%)": res["confidence"],
                        "Error": "",
                    }
                )
            else:
                rows.append(
                    {
                        "#": r["index"] + 1,
                        "Input": preview,
                        "Prediction": "",
                        "Confidence (%)": None,
                        "Error": r["error"],
                    }
                )
        df = pd.DataFrame(rows)
        if s["succeeded"]:
            chart(ui.donut(["Fake News", "Real News"], [s["fake"], s["real"]], [P.fake, P.real], P))
        ui.html_table(df, pill_column="Prediction")
        st.download_button(
            "⬇️ Download batch results (CSV)",
            df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"batch_results_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
        )


def page_compare() -> None:
    st.title("⚖️ Model comparison")
    info = call(client.models)
    if info:
        loaded = [m for m in info["models"] if m["loaded"] and m["metrics"]]
        if loaded:
            st.markdown("#### Held-out test performance")
            chart(ui.metrics_chart(loaded, P))
            ui.html_table(
                pd.DataFrame(
                    [
                        {
                            "Model": m["name"],
                            **{
                                k.upper()
                                if k == "f1"
                                else k.capitalize(): f"{(m['metrics'].get(k) or 0) * 100:.2f}%"
                                for k in ("accuracy", "precision", "recall", "f1")
                            },
                            "ROC AUC": m["metrics"].get("roc_auc"),
                            "Train time (s)": m["metrics"].get("train_seconds"),
                            "Default": "★" if m["is_default"] else "",
                        }
                        for m in loaded
                    ]
                )
            )
            ds = info.get("dataset") or {}
            st.caption(
                f"Dataset: {ds.get('name', '?')} · {ds.get('rows', '?')} articles · "
                f"test split {ds.get('test_rows', '?')} · trained {info.get('trained_at', '?')}"
            )

    st.markdown("#### Compare on your own article")
    text = st.text_area("Article text", height=200, key="compare_text")
    if st.button("⚖️ Compare all models", type="primary", width="stretch"):
        if not text.strip():
            st.warning("Please paste an article first.")
            return
        with st.spinner("Running all models…"):
            resp = call(client.compare, text=text, explain=True, top_k=5)
        if not resp:
            return
        cls = "fake" if resp["consensus"] == "Fake News" else "real"
        st.markdown(
            f'<div class="fnd-verdict {cls}">'
            f'<div class="label">Consensus: {resp["consensus"]}</div>'
            f'<div class="meta">{resp["agreement"]:.0f}% of models agree · mean P(fake) = '
            f"{resp['mean_fake_probability'] * 100:.1f}%</div></div>",
            unsafe_allow_html=True,
        )
        chart(ui.compare_chart(resp["results"], P))
        cols = st.columns(len(resp["results"]))
        for col, r in zip(cols, resp["results"], strict=True):
            with col, st.container(border=True):
                st.markdown(f"**{r['model_name']}**")
                st.markdown(
                    f"{'❌' if r['prediction'] == 'Fake News' else '✅'} {r['prediction']}"
                    f" · **{r['confidence']:.1f}%**"
                )
                exp = r.get("explanation")
                if exp:
                    top = exp["toward_fake"] + exp["toward_real"]
                    top = sorted(top, key=lambda c: -c["share"])[:5]
                    st.caption(
                        "Top signals: "
                        + ", ".join(f"{'🔴' if c['weight'] < 0 else '🟢'} {c['term']}" for c in top)
                    )


def page_history() -> None:
    st.title("📜 History")
    tab_session, tab_server = st.tabs(["This session", "All predictions (server)"])
    with tab_session:
        hist = st.session_state.history
        if not hist:
            st.info("No predictions in this session yet.")
        else:
            c1, c2, _ = st.columns([1, 1, 3])
            c1.download_button(
                "⬇️ Export CSV",
                to_csv_bytes(hist),
                file_name=f"fake_news_history_{datetime.now():%Y%m%d_%H%M%S}.csv",
                mime="text/csv",
                width="stretch",
            )
            if c2.button("🗑️ Clear", width="stretch"):
                st.session_state.history = []
                st.rerun()
            ui.html_table(to_dataframe(hist), pill_column="Prediction")
    with tab_server:
        rows = call(client.history, 200)
        if rows is not None:
            if rows:
                csv_bytes = call(client.history_csv)
                if csv_bytes:
                    st.download_button(
                        "⬇️ Export full server history (CSV)",
                        csv_bytes,
                        file_name="server_history.csv",
                        mime="text/csv",
                    )
                df = pd.DataFrame(rows)[
                    ["created_at", "source", "model", "label", "confidence", "text_preview"]
                ]
                df.columns = [
                    "Time (UTC)",
                    "Source",
                    "Model",
                    "Prediction",
                    "Confidence (%)",
                    "Preview",
                ]
                ui.html_table(df, pill_column="Prediction")
            else:
                st.info("The server has not recorded any predictions yet.")


def page_about() -> None:
    st.title("ℹ️ About")
    with st.container(border=True):
        st.markdown(
            "### 📌 Overview\n"
            "Text is cleaned, converted to **TF-IDF** unigram + bigram features, and classified "
            "by three models trained on the same split:\n\n"
            "| Model | Strength |\n|---|---|\n"
            "| **Random Forest** | Captures non-linear term interactions |\n"
            "| **Logistic Regression** | Fast, well-calibrated, fully interpretable |\n"
            "| **Linear SVM** (calibrated) | Strong margin-based text classifier |\n"
        )
    with st.container(border=True):
        st.markdown(
            "### 🔬 How explanations work\n"
            "- **Linear models:** each term's contribution = TF-IDF value × model coefficient.\n"
            "- **Random Forest:** the change in predicted probability at every split along each "
            "tree's decision path is credited to the split's term, then averaged over all trees "
            "(Saabas decomposition). Terms marked *(absent)* influenced the decision by **not** "
            "appearing in the text."
        )
    with st.container(border=True):
        st.markdown(
            "### ⚠️ Limitations\n"
            "These models learn *writing style* from their training data, not facts. A "
            "well-written false story can look 'real', and satire or opinion can look 'fake'. "
            "Use the output as one signal alongside source checking and fact-checking sites."
        )


PAGES = {
    "Home": page_home,
    "Analyze": page_analyze,
    "Batch": page_batch,
    "Compare models": page_compare,
    "History": page_history,
    "About": page_about,
}
PAGES[page]()

"""
Streamlit Frontend — Fake News Detection System v2.0
Features: Single analysis, Batch analysis, URL scraping, Model comparison, History export
"""
import streamlit as st
import requests
import pandas as pd
import json
from datetime import datetime

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.real-box { background: #d4edda; border-left: 5px solid #28a745; padding: 1rem; border-radius: 6px; }
.fake-box { background: #f8d7da; border-left: 5px solid #dc3545; padding: 1rem; border-radius: 6px; }
.stat-card { background: #f8f9fa; border-radius: 8px; padding: 1rem; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def call_predict(text: str, model: str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict", json={"news": text, "model": model}, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"API Error: {r.json().get('detail', 'Unknown')}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def call_predict_url(url: str, model: str) -> dict | None:
    try:
        r = requests.post(f"{API_URL}/predict/url", json={"url": url, "model": model}, timeout=30)
        if r.status_code == 200:
            return r.json()
        st.error(f"API Error: {r.json().get('detail', 'Unknown')}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def call_batch(articles: list, model: str) -> list | None:
    try:
        r = requests.post(f"{API_URL}/predict/batch", json={"articles": articles, "model": model}, timeout=60)
        if r.status_code == 200:
            return r.json()["results"]
        st.error(f"API Error: {r.json().get('detail', 'Unknown')}")
    except Exception as e:
        st.error(f"Connection error: {e}")
    return None

def call_stats() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/stats", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def show_result(result: dict, label: str = ""):
    pred = result["prediction"]
    conf = result.get("confidence")
    model = result.get("model_used", "unknown")
    if pred == "Real News":
        st.markdown(f'<div class="real-box">✅ <b>{pred}</b>{f" — {label}" if label else ""}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="fake-box">❌ <b>{pred}</b>{f" — {label}" if label else ""}</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Confidence", f"{conf}%" if conf else "N/A")
    c2.metric("Model", model.replace("_", " ").title())
    c3.metric("Words", result.get("word_count", "N/A"))
    if result.get("top_features"):
        with st.expander("🔍 Top Features"):
            feat_df = pd.DataFrame(result["top_features"].items(), columns=["Feature", "TF-IDF Score"])
            st.dataframe(feat_df, use_container_width=True)
    if result.get("note"):
        st.info(result["note"])


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 Fake News Detector")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "🔍 Analyze", "🔗 URL Check", "📦 Batch", "⚖️ Compare Models", "📊 Stats", "ℹ️ About"])
    st.markdown("---")
    model = st.selectbox("Model", ["random_forest", "logistic", "svm"],
                         format_func=lambda x: {"random_forest": "Random Forest", "logistic": "Logistic Regression", "svm": "SVM"}[x])
    st.markdown("---")
    health = api_health()
    if health:
        st.success("🟢 Backend connected")
    else:
        st.error("🔴 Backend offline")


# ── Pages ─────────────────────────────────────────────────────────────────────

if "Home" in page:
    st.title("🔍 AI Fake News Detection System v2.0")
    st.markdown("Detect fake news using **machine learning** — paste text, enter a URL, or analyze in bulk.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Models Available", "3")
    col2.metric("Batch Size", "Up to 20")
    col3.metric("Backend", "✅ Online" if health else "❌ Offline")
    st.markdown("---")
    st.markdown("""
    ### Features
    - **Single Analysis** — Paste any article and get instant prediction
    - **URL Scraping** — Enter a news URL and we fetch + analyze it
    - **Batch Analysis** — Check up to 20 articles at once
    - **Model Comparison** — See how 3 different ML models compare
    - **History Export** — Download your prediction history as CSV
    """)

elif "Analyze" in page:
    st.title("🔍 Analyze News Article")
    text = st.text_area("Paste your news article here", height=300, placeholder="Paste the full article text...")
    if st.button("Analyze", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                result = call_predict(text, model)
            if result:
                show_result(result)
                st.session_state.history.append({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "preview": text[:80] + "...",
                    "prediction": result["prediction"],
                    "confidence": result.get("confidence"),
                    "model": model,
                })
    st.markdown("---")
    st.subheader("📜 History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("⬇️ Export CSV", csv, "predictions.csv", "text/csv")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet.")

elif "URL" in page:
    st.title("🔗 Analyze from URL")
    url = st.text_input("Enter news article URL", placeholder="https://example.com/news/article")
    if st.button("Fetch & Analyze", type="primary", use_container_width=True):
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Fetching and analyzing..."):
                result = call_predict_url(url, model)
            if result:
                show_result(result)
                if result.get("scraped_text_preview"):
                    with st.expander("📄 Scraped Text Preview"):
                        st.write(result["scraped_text_preview"])

elif "Batch" in page:
    st.title("📦 Batch Analysis")
    st.info("Enter one article per line, or separate with `---`")
    raw = st.text_area("Articles (separate with ---)", height=400,
                       placeholder="Article 1 text here\n---\nArticle 2 text here")
    if st.button("Analyze All", type="primary", use_container_width=True):
        articles = [a.strip() for a in raw.split("---") if a.strip()]
        if not articles:
            st.warning("Please enter at least one article.")
        elif len(articles) > 20:
            st.error("Maximum 20 articles per batch.")
        else:
            with st.spinner(f"Analyzing {len(articles)} articles..."):
                results = call_batch(articles, model)
            if results:
                st.success(f"Done! Analyzed {len(results)} articles.")
                rows = []
                for i, r in enumerate(results):
                    pred = r["prediction"]
                    emoji = "✅" if pred == "Real News" else "❌"
                    rows.append({"#": i+1, "Result": f"{emoji} {pred}", "Confidence": f"{r.get('confidence', 'N/A')}%", "Words": r.get("word_count", "N/A")})
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

elif "Compare" in page:
    st.title("⚖️ Compare Models")
    text = st.text_area("Article to compare across all models", height=250)
    if st.button("Run All Models", type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter text.")
        else:
            models = ["random_forest", "logistic", "svm"]
            cols = st.columns(3)
            for col, m in zip(cols, models):
                with col:
                    st.subheader(m.replace("_", " ").title())
                    with st.spinner():
                        r = call_predict(text, m)
                    if r:
                        pred = r["prediction"]
                        conf = r.get("confidence", "N/A")
                        if pred == "Real News":
                            st.success(f"✅ {pred}")
                        else:
                            st.error(f"❌ {pred}")
                        st.metric("Confidence", f"{conf}%")

elif "Stats" in page:
    st.title("📊 API Statistics")
    stats = call_stats()
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Predictions", stats["total_predictions"])
        c2.metric("Fake News", stats["fake_count"])
        c3.metric("Real News", stats["real_count"])
        st.markdown("### Model Accuracies")
        acc_data = [(k, v["accuracy"] * 100, "✅" if v["loaded"] else "⚠️ Not loaded")
                    for k, v in stats["model_accuracies"].items()]
        st.dataframe(pd.DataFrame(acc_data, columns=["Model", "Accuracy %", "Status"]), use_container_width=True)
    else:
        st.error("Could not load stats. Is the backend running?")

elif "About" in page:
    st.title("ℹ️ About")
    st.markdown("""
    ### Fake News Detection System v2.0
    An ML-powered system to detect fake news articles.

    **Models:** Random Forest · Logistic Regression · SVM
    **Vectorization:** TF-IDF
    **Backend:** FastAPI
    **Frontend:** Streamlit

    **New in v2.0:**
    - URL scraping support
    - Batch analysis (up to 20 articles)
    - Multi-model comparison
    - Feature breakdown visualization
    - History export to CSV
    - API stats dashboard
    """)

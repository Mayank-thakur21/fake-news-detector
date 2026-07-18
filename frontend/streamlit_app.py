import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

if "history" not in st.session_state:
    st.session_state.history = []


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/", timeout=5)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def analyze_news(news_text: str) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"news": news_text},
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"API Error: {r.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Backend unavailable. Ensure FastAPI server is running.")
        return None


st.sidebar.title("📰 AI Fake News Detection")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Home", "Detection", "About"])

if page == "Home":
    st.title("🔍 AI Fake News Detection System")
    st.markdown(
        "This application uses Machine Learning to determine whether a "
        "news article is **Fake** or **Real**."
    )

    with st.container(border=True):
        st.markdown("### 📖 How to Use")
        st.markdown("""
        1. Navigate to the **Detection** page using the sidebar.
        2. Paste a news article into the text area.
        3. Click **Analyze News** to get the prediction.
        4. View your **Prediction History** on the same page.
        """)

    api_ok = check_api_health()
    if api_ok:
        st.success("✅ Backend is connected and running.")
    else:
        st.error("❌ Backend is not responding. Start the FastAPI server.")

elif page == "Detection":
    st.title("📰 News Analysis")

    news_text = st.text_area(
        "Paste News Article", height=300, placeholder="Paste full news article here..."
    )

    if st.button("🔍 Analyze News", type="primary", use_container_width=True):
        if not news_text.strip():
            st.warning("Please enter a news article before analyzing.")
        else:
            with st.spinner("🔎 Analyzing..."):
                result = analyze_news(news_text)

            if result:
                prediction = result["prediction"]
                confidence = result.get("confidence")

                if prediction == "Real News":
                    st.success(f"✅ {prediction}")
                else:
                    st.error(f"❌ {prediction}")

                if confidence is not None:
                    st.metric("Confidence", f"{confidence}%")

                preview = (
                    news_text[:80] + "..."
                    if len(news_text) > 80
                    else news_text
                )
                st.session_state.history.insert(
                    0, {"preview": preview, "prediction": prediction}
                )

    st.markdown("---")
    st.markdown("### 📜 Prediction History")

    if st.session_state.history:
        hist_data = [
            {"News Preview": h["preview"], "Prediction": h["prediction"]}
            for h in st.session_state.history
        ]
        st.table(hist_data)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions made yet.")

elif page == "About":
    st.title("ℹ️ About")

    with st.container(border=True):
        st.markdown("### 📌 Project Overview")
        st.markdown(
            "This is a **Fake News Detection System** that uses a "
            "**Random Forest Classifier** trained on a labeled dataset of "
            "real and fake news articles. The model analyzes the text and "
            "predicts whether it is likely to be fake or real."
        )

    with st.container(border=True):
        st.markdown("### 🛠️ Technologies Used")
        st.markdown("""
        - **Streamlit** – Interactive frontend
        - **FastAPI** – Backend API server
        - **scikit-learn** – Machine learning model & TF-IDF vectorizer
        - **joblib** – Model serialization
        - **pandas / numpy** – Data processing
        """)

    with st.container(border=True):
        st.markdown("### 📊 Dataset")
        st.markdown(
            "The model was trained on a combined dataset of real and fake "
            "news articles using **TF-IDF vectorization** and a "
            "**Random Forest** classifier, achieving high accuracy."
        )

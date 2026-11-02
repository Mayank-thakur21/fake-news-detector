AI Agent Prompt
Project Title

AI-Powered Fake News Detection System

Objective

Build a complete Fake News Detection web application using a pre-trained Machine Learning model.

Important:

The machine learning model has already been trained.
I will provide the following files:
fake_news_model.pkl
tfidf_vectorizer.pkl (or whatever the vectorizer file is named)
Do not retrain the model.
Load the saved model and vectorizer directly for inference.

The application should allow users to enter a news article and determine whether it is Fake News or Real News.

The project should be simple, clean, beginner-friendly, and suitable for a college mini project.

Tech Stack
Frontend
Streamlit
Backend
FastAPI
Uvicorn
Machine Learning
Python
scikit-learn
joblib
pandas
numpy
Project Structure
fake-news-detection/

│
├── backend/
│   ├── app.py
│   ├── predictor.py
│   ├── fake_news_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── requirements.txt
│
├── frontend/
│   └── streamlit_app.py
│
├── README.md
│
└── .gitignore
Backend Requirements

Load

fake_news_model.pkl

and

tfidf_vectorizer.pkl

during application startup.

Do not retrain anything.

API Endpoints
GET /

Return

{
    "message":"Fake News Detection API Running"
}
POST /predict

Accept JSON

{
    "news":"Paste complete news article here..."
}

Process

Clean text if necessary.
Transform using the loaded TF-IDF vectorizer.
Predict using the loaded model.
Return prediction.

Example response

{
    "prediction":"Fake News"
}

or

{
    "prediction":"Real News"
}

If the model supports probability (predict_proba), also return:

{
    "prediction":"Fake News",
    "confidence":96.4
}

Otherwise, omit the confidence score.

Streamlit Frontend

Use a modern and clean layout.

Use

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide"
)
Sidebar

Show

AI Fake News Detection

Navigation

• Home
• Detection
• About
Home Page

Display

AI Fake News Detection System

Description

"This application uses Machine Learning to determine whether a news article is Fake or Real."

Show an information box explaining how to use the application.

Detection Page

Create one large text area.

st.text_area(
    "Paste News Article",
    height=300
)

Below it place

Analyze News

button.

While prediction is running

Display

Analyzing...

using

st.spinner()
Prediction Result

If Real

Display

st.success("✅ Real News")

If Fake

Display

st.error("❌ Fake News")

If confidence is available

Display

st.metric(
    "Confidence",
    "96.5%"
)
Prediction History

Maintain prediction history using Streamlit session state.

Display a table containing

| News Preview | Prediction |

Only keep history for the current session.

No database is required.

About Page

Display

Project Overview

Technologies Used

Streamlit
FastAPI
Scikit-learn
TF-IDF
Random Forest (or whichever model was trained)

Dataset

Brief explanation of Fake News Detection.

UI Theme

Use

Wide layout
Sidebar
Cards using containers
Streamlit metrics
Professional spacing
Blue accent colors (default Streamlit theme is acceptable)
Emoji icons where appropriate

The interface should look modern and polished.

Error Handling

Handle

Empty input
Backend unavailable
Invalid requests
Missing model file
Missing vectorizer file

Display user-friendly messages.

README

Generate a professional README including

Project Overview
Features
Technologies
Folder Structure
Installation
Running Backend
Running Frontend
API Documentation
Future Improvements
Requirements

Generate a complete requirements.txt.

Code Quality
Modular architecture
Well-commented code
Type hints where appropriate
Reusable helper functions
PEP8 compliant
Running the Project

The project should run using

# Install dependencies
pip install -r backend/requirements.txt

# Start FastAPI
uvicorn backend.app:app --reload

# Start Streamlit
streamlit run frontend/streamlit_app.py
Important Instructions
Do NOT train any machine learning model.
Do NOT create a new TF-IDF vectorizer.
Load the existing .pkl files only.
The Streamlit frontend should communicate with the FastAPI backend using HTTP requests.
Write clean, production-style code that is easy to understand and maintain.
Ensure the application is fully functional after placing the provided .pkl files in the backend/ folder.
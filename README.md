# 🎬 Netflix Content Analyzer & Recommendation System

An end-to-end ML-powered Netflix content analysis and recommendation system built using Python, Pandas, and scikit-learn, and deployed as an interactive Streamlit web application.

This project demonstrates how machine learning can be applied to real-world content recommendation problems with explainable outputs.

---

## 🚀 Features

- Data cleaning & preprocessing pipeline
- Exploratory data analysis (EDA)
- Feature engineering (IMDb, runtime, binge score)
- Content-based recommendation system
- Categorical + numerical similarity modeling
- Explainable recommendations (“Why this show?”)
- Interactive Streamlit dashboard

---

## 🧠 Machine Learning Approach

- Type: Content-based recommendation system
- Similarity Metric: Cosine similarity
- Features Used:
  - Numerical: IMDb rating, runtime, seasons, binge score
  - Categorical: Genre, language (OneHotEncoded)
- Pipeline: ColumnTransformer + StandardScaler

---

## 🖥️ Streamlit Application

Users can:
- Select a Netflix show they like
- Get similar show recommendations
- See human-readable explanations for each recommendation

Example explanation:
- Same genre (Sci-Fi)
- Higher or similar IMDb rating
- Similar episode runtime
- High binge-watch score

---

## 📁 Project Structure

netflix-content-analyzer/
│
├── app/
│ └── streamlit_app.py
│
├── data/
│ ├── netflix_raw.csv
│ └── netflix_clean.csv
│
├── notebooks/
│ ├── 01_data_understanding.ipynb
│ ├── 02_eda.ipynb
│ └── 03_recommendation.ipynb
│
├── outputs/
│ └── *.png
│
├── src/
│ ├── cleaning.py
│ ├── analysis.py
│ ├── recommender.py
│ └── visualize.py
│
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/netflix-content-analyzer.git
cd netflix-content-analyzer

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python -m streamlit run app/streamlit_app.py


## 📸 Screenshots

### Dashboard
![Dashboard](screenshots/dashboard.png)

### ML-Powered Recommendations with Explanations
![Recommendations](screenshots/recommendations.png)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Path handling
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "netflix_clean.csv")

def load_data():
    return pd.read_csv(DATA_PATH)

def prepare_features(df):
    numeric_features = ["imdb", "runtime", "seasons", "binge_score"]
    categorical_features = ["main_genre", "language"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor.fit_transform(df)

def recommend(show_title, top_n=5):
    df = load_data()

    if show_title not in df["title"].values:
        return None

    features = prepare_features(df)
    similarity_matrix = cosine_similarity(features)

    idx = df[df["title"] == show_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_matches = similarity_scores[1:top_n+1]
    recommended_indices = [i[0] for i in top_matches]
    
    recommended_df = df.loc[recommended_indices].copy()
    recommended_df = df.loc[recommended_indices].copy()

    recommended_df["explanation"] = recommended_df.apply(
      lambda row: explain_recommendation(show_title, row, df),
      axis=1
    )

    return recommended_df[
       ["title", "main_genre", "imdb", "runtime", "explanation"]
    ]



def explain_recommendation(selected_show, recommended_row, original_df):
    base = original_df[original_df["title"] == selected_show].iloc[0]

    reasons = []

    if base["main_genre"] == recommended_row["main_genre"]:
        reasons.append(f"Same genre ({base['main_genre']})")

    if base["language"] == recommended_row["language"]:
        reasons.append(f"Same language ({base['language']})")

    if recommended_row["imdb"] >= base["imdb"]:
        reasons.append("Higher or similar IMDb rating")

    if abs(recommended_row["runtime"] - base["runtime"]) <= 5:
        reasons.append("Similar episode runtime")

    if recommended_row["binge_score"] >= base["binge_score"]:
        reasons.append("High binge-watch score")

    return reasons

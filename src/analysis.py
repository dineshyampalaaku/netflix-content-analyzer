import pandas as pd
import os

# Get absolute path to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLEAN_PATH = os.path.join(PROJECT_ROOT, "data", "netflix_clean.csv")


def load_data():
    return pd.read_csv(CLEAN_PATH)


def genre_distribution(df):
    result = df["main_genre"].value_counts().reset_index()
    result.columns = ["genre", "count"]
    return result


def language_distribution(df):
    return (
        df["language"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "language", "language": "count"})
    )


def top_imdb_shows(df, n=10):
    return df.sort_values("imdb", ascending=False).head(n)


def top_binge_shows(df, n=10):
    return df.sort_values("binge_score", ascending=False).head(n)


def recent_vs_old(df):
    return (
        df["is_recent"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "category", "is_recent": "count"})
    )


def avg_imdb_by_genre(df):
    return (
        df.groupby("main_genre")["imdb"]
        .mean()
        .round(2)
        .reset_index()
        .sort_values("imdb", ascending=False)
    )


if __name__ == "__main__":
    df = load_data()
    print("Data loaded:", df.shape)

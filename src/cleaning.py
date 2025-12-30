import pandas as pd
import numpy as np

RAW_PATH = "data/netflix_raw.csv"
CLEAN_PATH = "data/netflix_clean.csv"


def clean_netflix_data():
    # 1. Load raw data
    df = pd.read_csv(RAW_PATH)

    # 2. Drop duplicates
    df = df.drop_duplicates()

    # 3. Drop rows with missing critical values
    df = df.dropna(subset=["title", "genre", "imdb"])

    # 4. Normalize text columns
    df["title"] = df["title"].str.strip().str.title()
    df["genre"] = df["genre"].str.strip().str.title()
    df["language"] = df["language"].str.strip().str.title()

    # 5. Convert data types safely
    df["imdb"] = pd.to_numeric(df["imdb"], errors="coerce")
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    df["seasons"] = pd.to_numeric(df["seasons"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # 6. Remove invalid values
    df = df[(df["imdb"] >= 5.0) & (df["imdb"] <= 10.0)]
    df = df[df["runtime"] >= 20]
    df = df[df["seasons"] >= 1]
    df = df[df["year"] >= 2000]

    # 7. Feature engineering
    df["main_genre"] = df["genre"].apply(lambda x: x.split(",")[0])
    df["binge_score"] = (df["imdb"] / df["runtime"]).round(3)
    df["is_recent"] = df["year"].apply(lambda y: "Recent" if y >= 2020 else "Old")

    # 8. Reset index
    df = df.reset_index(drop=True)

    # 9. Save cleaned data
    df.to_csv(CLEAN_PATH, index=False)

    print("✅ Cleaning complete")
    print("📁 Saved:", CLEAN_PATH)
    print("📊 Final shape:", df.shape)


if __name__ == "__main__":
    clean_netflix_data()

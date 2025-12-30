import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "netflix_clean.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def plot_genre_distribution(df):
    data = df["main_genre"].value_counts()

    plt.figure()
    data.plot(kind="bar")
    plt.title("Genre Distribution on Netflix")
    plt.xlabel("Genre")
    plt.ylabel("Number of Shows")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "genre_distribution.png"))
    plt.close()


def plot_language_distribution(df):
    data = df["language"].value_counts()

    plt.figure()
    data.plot(kind="bar")
    plt.title("Language Distribution on Netflix")
    plt.xlabel("Language")
    plt.ylabel("Number of Shows")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "language_distribution.png"))
    plt.close()


def plot_imdb_distribution(df):
    plt.figure()
    df["imdb"].plot(kind="hist", bins=15)
    plt.title("IMDb Rating Distribution")
    plt.xlabel("IMDb Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "imdb_distribution.png"))
    plt.close()


def plot_top_binge_shows(df, n=10):
    top_binge = df.sort_values("binge_score", ascending=False).head(n)

    plt.figure()
    plt.barh(top_binge["title"], top_binge["binge_score"])
    plt.xlabel("Binge Score")
    plt.title("Top 10 Binge-Worthy Shows")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "top_binge_shows.png"))
    plt.close()


def plot_avg_imdb_by_genre(df):
    avg_imdb = df.groupby("main_genre")["imdb"].mean().sort_values(ascending=False)

    plt.figure()
    avg_imdb.plot(kind="bar")
    plt.title("Average IMDb Rating by Genre")
    plt.xlabel("Genre")
    plt.ylabel("Average IMDb")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "avg_imdb_by_genre.png"))
    plt.close()


if __name__ == "__main__":
    df = load_data()

    plot_genre_distribution(df)
    plot_language_distribution(df)
    plot_imdb_distribution(df)
    plot_top_binge_shows(df)
    plot_avg_imdb_by_genre(df)

    print("✅ All plots generated and saved in /outputs")

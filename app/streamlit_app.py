import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import streamlit as st
import pandas as pd
import os
from src.recommender import recommend

# Path handling
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "netflix_clean.csv")
#from src.recommender import recommend
# Load data
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.title("🎬 Netflix Content Analyzer")
st.markdown("Interactive analysis of Netflix shows using Python & Pandas")

# Sidebar filters
st.sidebar.header("Filters")

genre_filter = st.sidebar.selectbox(
    "Select Genre",
    ["All"] + sorted(df["main_genre"].unique())
)

language_filter = st.sidebar.selectbox(
    "Select Language",
    ["All"] + sorted(df["language"].unique())
)

filtered_df = df.copy()

if genre_filter != "All":
    filtered_df = filtered_df[filtered_df["main_genre"] == genre_filter]

if language_filter != "All":
    filtered_df = filtered_df[filtered_df["language"] == language_filter]

# Main content
st.subheader("📊 Filtered Shows")
st.dataframe(filtered_df)

st.subheader("⭐ Top IMDb Rated Shows")
st.dataframe(filtered_df.sort_values("imdb", ascending=False).head(10))

st.subheader("🍿 Most Binge-Worthy Shows")
st.dataframe(filtered_df.sort_values("binge_score", ascending=False).head(10))

st.subheader("📈 Quick Stats")
col1, col2, col3 = st.columns(3)

col1.metric("Total Shows", len(filtered_df))
col2.metric("Average IMDb", round(filtered_df["imdb"].mean(), 2))
col3.metric("Avg Runtime", round(filtered_df["runtime"].mean(), 1))

st.divider()
st.subheader("🤖 ML-Powered Recommendations")

show_list = sorted(df["title"].unique())

selected_show = st.selectbox(
    "Select a show you like",
    show_list
)

if st.button("Recommend Similar Shows"):
    recommendations = recommend(selected_show)

    if recommendations is None or recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.success(f"Because you liked **{selected_show}**, you may also like:")
        for _, row in recommendations.iterrows():
            st.markdown(f"### 🎬 {row['title']}")
            st.write(f"**Genre:** {row['main_genre']}")
            st.write(f"**IMDb:** {row['imdb']} | **Runtime:** {row['runtime']} min")

            st.write("**Why recommended:**")
            for reason in row["explanation"]:
                st.write(f"• {reason}")

            st.divider()


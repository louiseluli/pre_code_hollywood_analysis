import streamlit as st
import pandas as pd
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Pre-Code Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- Title and Description ---
st.title("🎬 Hybrid Movie Recommender")
st.write(
    "Enter a movie title to get recommendations based on both content similarity "
    "(what the movie is about) and collaborative filtering (what similar users liked)."
)

# --- Placeholder Data Loading ---
# In a real app, we would load our saved models and data here.
# For now, let's create dummy data to build the UI.
@st.cache_data
def load_data():
    # This function would load movie titles, similarity matrix, etc.
    # Dummy data:
    movie_titles = ["The Public Enemy", "Frankenstein", "Dracula", "Little Caesar", "City Lights", "The Old Dark House"]
    return movie_titles

movie_titles = load_data()

# --- User Input ---
st.header("Find Movies You'll Love")
selected_movie = st.selectbox(
    "Start by selecting a movie you like:",
    options=movie_titles
)

# --- Recommendation Logic (Placeholder) ---
def get_recommendations(movie_title, num_recs=5):
    # This function will contain our hybrid ML logic.
    # For now, it just returns some dummy recommendations.
    recs = {
        "title": ["The Kiss Before the Mirror", "Waterloo Bridge", "The Impatient Maiden", "By Candlelight", "One More River"],
        "poster_path": ["/wB4dnB5sFpS2k2n4L2tL1e0iA5g.jpg", "/iTQpU3oA2k1a2T5pB3A2t5pB3A.jpg", "/pU3oA2k1a2T5pB3A2t5pB3A.jpg", "/k1a2T5pB3A2t5pB3A2t5pB3.jpg", "/a2T5pB3A2t5pB3A2t5pB3A.jpg"]
    }
    return pd.DataFrame(recs)


# --- Display Recommendations ---
if st.button("Get Recommendations", type="primary"):
    if selected_movie:
        st.subheader(f"Because you liked '{selected_movie}', you might also like...")

        recommendations_df = get_recommendations(selected_movie)

        # Create columns to display recommendations side-by-side
        cols = st.columns(5)
        for i, row in recommendations_df.iterrows():
            with cols[i]:
                st.image(f"https://image.tmdb.org/t/p/w200{row['poster_path']}", use_column_width=True)
                st.caption(f"**{row['title']}**")
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from annoy import AnnoyIndex
from surprise import SVD

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Explorer & Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- Data and Model Loading (with Caching)---
@st.cache_resource
def load_models():
    """Loads the Annoy index and the SVD model from disk."""
    with open("data/collaborative_model.pkl", 'rb') as f:
        svd_model = pickle.load(f)
    annoy_index = AnnoyIndex(384, 'angular')
    annoy_index.load("data/movie_content_index.ann")
    return svd_model, annoy_index

@st.cache_data
def load_data():
    """Loads all necessary dataframes and creates mappings."""
    index_to_tconst = pd.read_pickle("data/index_to_tconst.pkl")
    enriched_df = pd.read_pickle("../data/processed/hollywood_galaxy_df.pkl")
    enriched_df['primary_genre'] = enriched_df['genres'].fillna('Unknown').str.split(',').str[0]
    
    # NEW: Load the original titles data for more detailed display
    titles_df = pd.read_csv("../data/raw_imdb/title.basics.tsv", sep='\t', usecols=['tconst', 'primaryTitle', 'originalTitle', 'startYear'])
    title_details = titles_df.set_index('tconst')

    movies_in_model = enriched_df[enriched_df['tconst'].isin(index_to_tconst.values)]
    all_genres = sorted(list(enriched_df['primary_genre'].dropna().unique()))

    return index_to_tconst, enriched_df, movies_in_model, all_genres, title_details

# --- NEW: Add a loading spinner for a better user experience ---
with st.spinner('Loading recommendation models... This may take a moment.'):
    svd_model, annoy_index = load_models()
with st.spinner('Loading movie data...'):
    index_to_tconst, enriched_df, movies_in_model, all_genres, title_details = load_data()

# --- Sidebar for Filters ---
st.sidebar.title("🎬 Movie Explorer")
st.sidebar.header("Advanced Search Filters")
selected_genre = st.sidebar.multiselect("Filter by Genre(s):", all_genres)
actor_search = st.sidebar.text_input("Search for an Actor/Actress:")
keyword_search = st.sidebar.text_input("Search for a Keyword:")

# --- Main Application ---
st.title("Pre-Code Hollywood Movie Explorer & Recommender")

# --- Explorer Mode ---
with st.expander("Explorer Mode - Search the Database", expanded=True):
    # Filter logic remains the same
    filtered_df = enriched_df
    if selected_genre:
        filtered_df = filtered_df[filtered_df['primary_genre'].isin(selected_genre)]
    if actor_search:
        actor_df = pd.read_pickle("../data/processed/hollywood_df.pkl")
        actor_movies = actor_df[actor_df['primaryName'].str.contains(actor_search, case=False)]['tconst'].unique()
        filtered_df = filtered_df[filtered_df['tconst'].isin(actor_movies)]
    if keyword_search:
        filtered_df = filtered_df[filtered_df['keywords'].str.contains(keyword_search, case=False)]

    st.write(f"Found **{len(filtered_df)}** movies matching your criteria.")
    if not filtered_df.empty:
        num_cols = 5
        cols = st.columns(num_cols)
        for i, row in enumerate(filtered_df.head(15).itertuples()):
            with cols[i % num_cols]:
                if pd.notna(row.poster_path):
                    st.image(f"https://image.tmdb.org/t/p/w200{row.poster_path}", use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/200x300.png?text=No+Poster", use_column_width=True)
                
                # NEW: More detailed caption
                caption_text = f"**{row.primaryTitle}** ({int(row.startYear)})"
                if row.primaryTitle != row.originalTitle:
                    caption_text += f"<br><i>{row.originalTitle}</i>"
                st.caption(caption_text, unsafe_allow_html=True)

# --- Recommender Mode ---
st.header("Recommender Mode")
selected_movie_title = st.selectbox(
    "Select a movie you like to get recommendations:",
    options=sorted(movies_in_model['primaryTitle'].unique())
)

def get_recommendations(tconst, num_recs=5):
    title_series = pd.Series(index_to_tconst.index, index=index_to_tconst.values)
    movie_idx = title_series.get(tconst)
    if movie_idx is None: return []
    rec_indices = annoy_index.get_nns_by_item(movie_idx, num_recs + 1)[1:]
    return index_to_tconst[rec_indices].tolist()

if st.button("Get Recommendations", type="primary"):
    if selected_movie_title:
        selected_tconst = movies_in_model[movies_in_model['primaryTitle'] == selected_movie_title]['tconst'].values[0]
        st.subheader(f"Because you liked '{selected_movie_title}', you might also like...")
        
        rec_tconsts = get_recommendations(selected_tconst)
        
        if rec_tconsts:
            cols = st.columns(len(rec_tconsts))
            for i, tconst in enumerate(rec_tconsts):
                with cols[i]:
                    rec_info = title_details.loc[tconst]
                    poster_info = enriched_df.loc[enriched_df['tconst'] == tconst]
                    
                    poster_path = poster_info['poster_path'].iloc[0] if not poster_info.empty else None

                    if pd.notna(poster_path):
                        st.image(f"https://image.tmdb.org/t/p/w200{poster_path}", use_column_width=True)
                    else:
                        st.image("https://via.placeholder.com/200x300.png?text=No+Poster", use_column_width=True)
                    
                    # NEW: More detailed caption for recommendations
                    caption_text = f"**{rec_info.primaryTitle}** ({int(rec_info.startYear)})"
                    if rec_info.primaryTitle != rec_info.originalTitle:
                        caption_text += f"<br><i>{rec_info.originalTitle}</i>"
                    st.caption(caption_text, unsafe_allow_html=True)
        else:
            st.warning("Sorry, we couldn't find any recommendations for this movie.")
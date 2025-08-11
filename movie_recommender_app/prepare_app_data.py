# movie_recommender_app/prepare_app_data.py

import pandas as pd
import numpy as np
import os
import json
import tmdbsimple as tmdb
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from dotenv import load_dotenv
from tqdm import tqdm

print("--- Starting Full Data Preparation Pipeline for Streamlit App ---")

# --- 1. SETUP: Load Environment Variables and Define Paths ---
load_dotenv()
tmdb.API_KEY = os.getenv('TMDB_API_KEY')
if not tmdb.API_KEY:
    print("FATAL: TMDB API key not found. Please check your .env file.")
    exit()

DATA_ROOT = "../data/"
RAW_IMDB_DIR = os.path.join(DATA_ROOT, "raw_imdb")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
CACHE_DIR = os.path.join(DATA_ROOT, "tmdb_cache")
FINAL_APP_DATA_PATH = os.path.join(PROCESSED_DIR, "app_data.pkl")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# --- 2. DATA LOADING: Get the list of Hollywood movie tconsts ---
print("Step 1: Loading initial Hollywood movie list...")
hollywood_df = pd.read_pickle(os.path.join(PROCESSED_DIR, "hollywood_df.pkl"))
hollywood_tconsts = set(hollywood_df['tconst'].unique())
titles_df = pd.read_csv(
    os.path.join(RAW_IMDB_DIR, "title.basics.tsv"),
    sep='\t',
    usecols=['tconst', 'primaryTitle', 'originalTitle', 'startYear', 'genres']
)
master_df = titles_df[titles_df['tconst'].isin(hollywood_tconsts)].copy()
print(f"Loaded details for {len(master_df)} Hollywood movies.")

# --- 3. DATA ENRICHMENT: Fetch Keywords from TMDB with Caching ---
print("\nStep 2: Enriching data with keywords from TMDB (using cache)...")

# --- THIS FUNCTION IS CORRECTED ---
def get_keywords_from_tmdb(tconst, cache_dir):
    cache_filepath = os.path.join(cache_dir, f"{tconst}.json")
    if os.path.exists(cache_filepath):
        with open(cache_filepath, 'r') as f:
            cached_data = json.load(f)
            # Always return a tuple for consistency
            return cached_data.get('keywords', ''), cached_data.get('poster_path', '')

    try:
        find = tmdb.Find(tconst)
        response = find.info(external_source='imdb_id')
        if not response['movie_results']:
            result_dict = {"keywords": "", "poster_path": ""}
        else:
            movie_id = response['movie_results'][0]['id']
            movie = tmdb.Movies(movie_id)
            keywords = movie.keywords()['keywords']
            result_dict = {
                "keywords": ' '.join([k['name'] for k in keywords]),
                "poster_path": response['movie_results'][0].get('poster_path', '')
            }
    except Exception:
        result_dict = {"keywords": "", "poster_path": ""}
    
    with open(cache_filepath, 'w') as f:
        json.dump(result_dict, f)
    
    # Always return a tuple for consistency
    return result_dict['keywords'], result_dict['poster_path']

tqdm.pandas(desc="Fetching TMDB Data")
tmdb_data = master_df['tconst'].progress_apply(lambda t: get_keywords_from_tmdb(t, CACHE_DIR))
# This assignment is now robust because the input is a list of consistent tuples
master_df[['keywords', 'poster_path']] = pd.DataFrame(tmdb_data.tolist(), index=master_df.index)

# --- 4. ML: Generate "Movie DNA" Embeddings with Caching ---
print("\nStep 3: Generating 'Movie DNA' embeddings (using cache)...")
EMBEDDINGS_PATH = os.path.join(PROCESSED_DIR, "app_embeddings.npy")
if os.path.exists(EMBEDDINGS_PATH):
    print("Found cached embeddings. Loading...")
    movie_embeddings = np.load(EMBEDDINGS_PATH)
else:
    print("Generating new embeddings...")
    corpus = master_df['keywords'].fillna('').tolist()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    movie_embeddings = model.encode(corpus, show_progress_bar=True)
    np.save(EMBEDDINGS_PATH, movie_embeddings)
    print("Embeddings saved.")

# --- 5. ML: Run t-SNE Dimensionality Reduction with Caching ---
print("\nStep 4: Running t-SNE for 2D mapping (using cache)...")
TSNE_PATH = os.path.join(PROCESSED_DIR, "app_tsne_coords.npy")
if os.path.exists(TSNE_PATH):
    print("Found cached t-SNE coordinates. Loading...")
    tsne_coords = np.load(TSNE_PATH)
else:
    print("Running new t-SNE calculation...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=42, verbose=1)
    tsne_coords = tsne.fit_transform(movie_embeddings)
    np.save(TSNE_PATH, tsne_coords)
    print("t-SNE coordinates saved.")

# --- 6. FINAL ASSEMBLY: Combine everything and save ---
print("\nStep 5: Assembling final data file for the app...")
master_df['x'] = tsne_coords[:, 0]
master_df['y'] = tsne_coords[:, 1]
master_df['primary_genre'] = master_df['genres'].fillna('Unknown').str.split(',').str[0]
master_df.to_pickle(FINAL_APP_DATA_PATH)
print(f"\n--- Pipeline Finished ---")
print(f"Canonical data file for the app saved to: {FINAL_APP_DATA_PATH}")
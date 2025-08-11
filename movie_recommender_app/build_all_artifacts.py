# movie_recommender_app/build_all_artifacts.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import numpy as np
import json
import tmdbsimple as tmdb
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from surprise import SVD, Dataset, Reader
import pickle
from dotenv import load_dotenv
from tqdm import tqdm
from annoy import AnnoyIndex

print("--- Starting Full Data & Model Artifact Pipeline ---")

# --- 1. SETUP ---
load_dotenv()
tmdb.API_KEY = os.getenv('TMDB_API_KEY')
if not tmdb.API_KEY:
    print("FATAL: TMDB API key not found. Please check your .env file.")
    exit()

DATA_ROOT = "../data/"
RAW_IMDB_DIR = os.path.join(DATA_ROOT, "raw_imdb")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")
CACHE_DIR = os.path.join(DATA_ROOT, "tmdb_cache")
APP_DATA_DIR = "data/"
FINAL_APP_DATA_PATH = os.path.join(PROCESSED_DIR, "app_data.pkl")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(APP_DATA_DIR, exist_ok=True)

# --- 2. DATA LOADING & FILTERING ---
print("\nStep 1: Loading and filtering the definitive movie list...")
titles_df = pd.read_csv(os.path.join(RAW_IMDB_DIR, "title.basics.tsv"), sep='\t', low_memory=False)
ratings_df = pd.read_csv(os.path.join(RAW_IMDB_DIR, 'title.ratings.tsv'), sep='\t', na_values='\\N')
movies_df = titles_df[titles_df['titleType'] == 'movie'].copy()
relevant_movies = ratings_df[ratings_df['numVotes'] > 1000]
master_df = movies_df[movies_df['tconst'].isin(relevant_movies['tconst'])].copy()
master_df.dropna(subset=['genres', 'runtimeMinutes'], inplace=True)
master_df = master_df.reset_index(drop=True)
print(f"Created a definitive set of {len(master_df)} movies.")

# --- 3. DATA ENRICHMENT (TMDB) ---
print("\nStep 2: Enriching with TMDB keywords (using cache)...")

# --- THIS FUNCTION IS CORRECTED ---
def get_tmdb_data(tconst, cache_dir):
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
            result_dict = {"keywords": ' '.join([k['name'] for k in keywords]), "poster_path": response['movie_results'][0].get('poster_path', '')}
    except Exception:
        result_dict = {"keywords": "", "poster_path": ""}
    with open(cache_filepath, 'w') as f:
        json.dump(result_dict, f)
    # Always return a tuple for consistency
    return result_dict['keywords'], result_dict['poster_path']

tmdb_results = [get_tmdb_data(t, CACHE_DIR) for t in tqdm(master_df['tconst'], desc="Fetching TMDB Data")]
master_df[['keywords', 'poster_path']] = pd.DataFrame(tmdb_results, index=master_df.index)
print("Finished enriching data.")

# --- 4. CONTENT-BASED MODEL ---
print("\nStep 3: Building Content-Based Model...")
corpus = master_df['genres'].str.replace(',', ' ') + ' ' + master_df['keywords'].fillna('')
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(corpus, show_progress_bar=True)
embedding_dim = embeddings.shape[1]
annoy_index = AnnoyIndex(embedding_dim, 'angular')
for i, vec in enumerate(embeddings): annoy_index.add_item(i, vec)
annoy_index.build(10)
annoy_index.save(os.path.join(APP_DATA_DIR, 'movie_content_index.ann'))
index_to_tconst = pd.Series(master_df['tconst'].values)
index_to_tconst.to_pickle(os.path.join(APP_DATA_DIR, 'index_to_tconst.pkl'))
print("Content-Based Model saved.")

# --- 5. COLLABORATIVE FILTERING MODEL ---
print("\nStep 4: Building Collaborative Filtering Model...")
ratings_subset = ratings_df[ratings_df['tconst'].isin(master_df['tconst'])].head(1000000)
ratings_subset['userID'] = range(len(ratings_subset))
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(ratings_subset[['userID', 'tconst', 'averageRating']], reader)
trainset = data.build_full_trainset()
svd = SVD(n_factors=50, n_epochs=20, random_state=42, verbose=True)
svd.fit(trainset)
with open(os.path.join(APP_DATA_DIR, 'collaborative_model.pkl'), 'wb') as f:
    pickle.dump(svd, f)
print("Collaborative Filtering Model saved.")

# --- 6. t-SNE & FINAL DATA ASSEMBLY ---
print("\nStep 5: Running t-SNE and assembling final app data...")
tsne = TSNE(n_components=2, perplexity=30, init='pca', n_iter=1000, random_state=42, verbose=1)
tsne_coords = tsne.fit_transform(embeddings)
master_df['x'] = tsne_coords[:, 0]
master_df['y'] = tsne_coords[:, 1]
master_df['primary_genre'] = master_df['genres'].str.split(',').str[0]
master_df.to_pickle(os.path.join(PROCESSED_DIR, "app_data.pkl"))
print("Final app_data.pkl saved.")

print("\n--- ALL ARTIFACTS BUILT SUCCESSFULLY ---")
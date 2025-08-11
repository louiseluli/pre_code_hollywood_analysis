import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_feature_engineering_pipeline(input_path, output_dir):
    """
    Loads the cleaned Hollywood data, creates a metadata soup, vectorizes it,
    computes the cosine similarity matrix, and saves the results.
    """
    print("--- Starting Feature Engineering Pipeline ---")

    # 1. Load Data
    print(f"Loading data from {input_path}...")
    df = pd.read_pickle(input_path)

    # 2. Prepare Data for Metadata Soup
    print("Aggregating cast and director data for each movie...")
    
    directors = df[df['category'] == 'director'].groupby('tconst')['primaryName'].apply(list)
    actors = df[df['category'].isin(['actor', 'actress'])] \
        .groupby('tconst')['primaryName'] \
        .apply(lambda x: list(x.head(5)))

    movies_df = df[['tconst', 'primaryTitle', 'genres']].drop_duplicates(subset=['tconst']).set_index('tconst')
    
    movies_df = movies_df.join(directors.rename('directors')).join(actors.rename('actors'))
    
    # 3. Create Metadata Soup
    print("Creating metadata 'soup' for each movie...")
    
    # CORRECTED FUNCTION TO HANDLE MISSING (NaN) VALUES
    def create_soup(x):
        # Use pd.notna() to check for missing values before trying string operations
        genres = x['genres'].replace(',', ' ') if pd.notna(x['genres']) else ''
        
        # The isinstance() check already handles missing director/actor lists correctly
        directors_str = ' '.join(x['directors']).replace(' ', '') if isinstance(x['directors'], list) else ''
        actors_str = ' '.join(x['actors']).replace(' ', '') if isinstance(x['actors'], list) else ''
        
        return f"{genres} {directors_str} {actors_str}"

    movies_df['soup'] = movies_df.apply(create_soup, axis=1)

    # 4. Vectorize the Soup using TF-IDF
    print("Vectorizing metadata using TF-IDF...")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['soup'])

    # 5. Calculate Cosine Similarity
    print("Calculating cosine similarity matrix...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # 6. Save the outputs
    os.makedirs(output_dir, exist_ok=True)
    
    sim_matrix_path = os.path.join(output_dir, 'cosine_similarity_matrix.npy')
    np.save(sim_matrix_path, cosine_sim)
    print(f"Similarity matrix saved to {sim_matrix_path}")

    indices_path = os.path.join(output_dir, 'movie_indices.pkl')
    indices = pd.Series(movies_df.index, index=movies_df['primaryTitle']).drop_duplicates()
    with open(indices_path, 'wb') as f:
        pickle.dump(indices, f)
    print(f"Movie indices saved to {indices_path}")

    print("--- Pipeline Finished ---")
    return movies_df, cosine_sim, indices


if __name__ == "__main__":
    HOLLYWOOD_DF_PATH = "data/processed/hollywood_df.pkl"
    OUTPUT_DATA_DIR = "data/processed"
    
    if os.path.exists(HOLLYWOOD_DF_PATH):
        create_feature_engineering_pipeline(HOLLYWOOD_DF_PATH, OUTPUT_DATA_DIR)
    else:
        print(f"Error: Processed data not found at {HOLLYWOOD_DF_PATH}")
        print("Please run the analysis notebook first to create and save 'hollywood_df'.")
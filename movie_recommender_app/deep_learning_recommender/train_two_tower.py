# deep_learning_recommender/train_two_tower.py

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

import tensorflow as tf
import tensorflow_recommenders as tfrs  # kept for future extensibility

print("--- Starting Two-Tower Model Training Pipeline ---")

# ========= 0) Robust paths & sanity checks =========
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[2]

DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))
PROCESSED_DIR = DATA_ROOT / "processed"
APP_DATA_DIR = PROJECT_ROOT / "movie_recommender_app" / "data"

print("Resolved paths:")
print(f"  PROJECT_ROOT  = {PROJECT_ROOT}")
print(f"  PROCESSED_DIR = {PROCESSED_DIR}")
print(f"  APP_DATA_DIR  = {APP_DATA_DIR}")

def require_file(path: Path, hint: str = "") -> Path:
    if not path.exists():
        msg = f"Missing required file: {path}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)
    return path

# ========= 1) Load pre-processed data =========
print("\nStep 1: Loading pre-processed data...")
app_data_pkl = require_file(PROCESSED_DIR / "app_data.pkl")
embeddings_npy = require_file(PROCESSED_DIR / "app_embeddings.npy",
                              hint="If this is for a larger corpus, we will align it.")
index_to_tconst_pkl = require_file(APP_DATA_DIR / "index_to_tconst.pkl")
master_df = pd.read_pickle(app_data_pkl)
embeddings = np.load(embeddings_npy)
index_to_tconst = pd.read_pickle(index_to_tconst_pkl)

print(f"  master_df shape      : {master_df.shape}")
print(f"  embeddings shape     : {embeddings.shape}")
print(f"  index_to_tconst size : {len(index_to_tconst)}")

num_movies = len(master_df)

# ========= 1b) ALIGN embeddings to app subset if needed =========
if embeddings.shape[0] != num_movies:
    print("\n[align] Embedding row count does not match app data. Attempting alignment via hollywood_df.pkl ...")
    hollywood_df_pkl = require_file(PROCESSED_DIR / "hollywood_df.pkl",
                                    hint="Should correspond to the big embeddings file row order.")
    hollywood_df = pd.read_pickle(hollywood_df_pkl)

    if hollywood_df.shape[0] != embeddings.shape[0]:
        raise ValueError(
            f"hollywood_df rows ({hollywood_df.shape[0]}) != embeddings rows ({embeddings.shape[0]}). "
            "Cannot infer alignment."
        )

    if "tconst" not in hollywood_df.columns:
        raise KeyError("Column 'tconst' not found in hollywood_df.pkl")

    # Build mapping from tconst -> row index in the big embeddings
    print("[align] Building tconst->row mapping...")
    tconst_to_row = {t: i for i, t in enumerate(hollywood_df["tconst"].astype(str).tolist())}

    # Ensure index_to_tconst is a list of strings
    idx_tconst_list = [str(t) for t in (index_to_tconst.tolist() if hasattr(index_to_tconst, "tolist") else list(index_to_tconst))]

    app_rows = []
    missing = []
    for t in idx_tconst_list:
        i = tconst_to_row.get(t)
        if i is None:
            missing.append(t)
        else:
            app_rows.append(i)

    if missing:
        print(f"[align] WARNING: {len(missing)} tconsts in app_data not found in hollywood_df. They will be dropped.")

    # Subset embeddings and app data to common items
    aligned_embeddings = embeddings[app_rows]
    keep_mask = [t not in set(missing) for t in idx_tconst_list]
    master_df = master_df.loc[keep_mask].reset_index(drop=True)
    idx_tconst_list = [t for t in idx_tconst_list if t not in set(missing)]

    # Persist aligned embeddings to speed up future runs
    aligned_path = PROCESSED_DIR / "app_embeddings_aligned.npy"
    np.save(aligned_path, aligned_embeddings)
    embeddings = aligned_embeddings

    print(f"[align] Done. Aligned embeddings shape: {embeddings.shape}")
    print(f"[align] master_df shape (after drop): {master_df.shape}")

    # Rebuild Annoy index to guarantee consistency with the aligned embeddings
    print("[align] Rebuilding Annoy index for aligned embeddings...")
    aligned_ann_path = APP_DATA_DIR / "movie_content_index.ann"
    aligned_ann_path.parent.mkdir(parents=True, exist_ok=True)
    ann = AnnoyIndex(embeddings.shape[1], "angular")
    for i, vec in enumerate(embeddings):
        ann.add_item(i, vec.tolist())
    ann.build(50)
    ann.save(str(aligned_ann_path))
    print(f"[align] Saved rebuilt Annoy index: {aligned_ann_path}")

# ========= 2) Generate training triplets (robust) =========
print("\nStep 2: Generating positive and negative training pairs...")
ann_path = require_file(APP_DATA_DIR / "movie_content_index.ann",
                        hint="Annoy index should match the aligned embeddings.")
annoy_index = AnnoyIndex(embeddings.shape[1], "angular")
annoy_index.load(str(ann_path))

if annoy_index.f != embeddings.shape[1]:
    raise ValueError(
        f"Annoy index dimension ({annoy_index.f}) != embeddings dim ({embeddings.shape[1]})."
    )

rng = np.random.default_rng(42)
anchors, positives, negatives = [], [], []
# NEW: also keep the row indices for eval later
anchor_ids, positive_ids, negative_ids = [], [], []
num_movies = len(master_df)

# Try Annoy first (wider K and search_k=-1), then fallback if needed.
successful = 0
for movie_idx in range(num_movies):
    # ask for a larger neighborhood, then drop 'self'
    neighbors = annoy_index.get_nns_by_item(movie_idx, 50, search_k=-1)
    neighbors = [j for j in neighbors if j != movie_idx]
    if neighbors:
        pos_idx = neighbors[0]
        neg_idx = movie_idx
        while neg_idx == movie_idx or neg_idx == pos_idx:
            neg_idx = int(rng.integers(0, num_movies))

        # vectors
        anchors.append(embeddings[movie_idx])
        positives.append(embeddings[pos_idx])
        negatives.append(embeddings[neg_idx])

        # indices
        anchor_ids.append(movie_idx)
        positive_ids.append(pos_idx)
        negative_ids.append(neg_idx)

        successful += 1


print(f"  Annoy-based triplets: {successful} / {num_movies}")

# Fallback: if Annoy didn’t produce anything (or produced too few), build pairs via cosine similarity
MIN_TRIPLETS = max(500, num_movies // 4)  # require at least some coverage
if successful < MIN_TRIPLETS:
    print("  [fallback] Using cosine-similarity to find positives (one per item).")
    # Ensure L2-normalized embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X = embeddings / norms
    # Similarity matrix (float32): ~84MB for 4.6k x 4.6k, OK
    sim = X @ X.T
    # Don’t choose itself
    np.fill_diagonal(sim, -np.inf)
    top_pos = np.argmax(sim, axis=1)  # (N,)

    # Reset lists (vectors + indices)
    anchors, positives, negatives = [], [], []
    anchor_ids, positive_ids, negative_ids = [], [], []

    for i in range(num_movies):
        pos_idx = int(top_pos[i])
        neg_idx = i
        while neg_idx == i or neg_idx == pos_idx:
            neg_idx = int(rng.integers(0, num_movies))

        # vectors
        anchors.append(embeddings[i])
        positives.append(embeddings[pos_idx])
        negatives.append(embeddings[neg_idx])

        # indices
        anchor_ids.append(i)
        positive_ids.append(pos_idx)
        negative_ids.append(neg_idx)

    print(f"  [fallback] Triplets built: {len(anchors)}")


# Ensure proper dtypes and shapes
anchors = np.asarray(anchors, dtype=np.float32)
positives = np.asarray(positives, dtype=np.float32)
negatives = np.asarray(negatives, dtype=np.float32)

assert anchors.ndim == positives.ndim == negatives.ndim == 2, \
    f"Bad shapes: {anchors.shape}, {positives.shape}, {negatives.shape}"
print(f"  Triplets final: {anchors.shape[0]} (dim={anchors.shape[1]})")

features = {"anchor": anchors, "positive": positives, "negative": negatives}
dataset = (
    tf.data.Dataset.from_tensor_slices(features)
    .shuffle(min(10_000, anchors.shape[0]), seed=123)
    .batch(128)
    .prefetch(tf.data.AUTOTUNE)
)

# Peek one batch
for batch in dataset.take(1):
    print("  Batch example shapes:", {k: v.shape for k, v in batch.items()})
    
# ========= 2b) Save triplets for future use =========
triplets_path = PROCESSED_DIR / "movie_triplets.npz"
np.savez_compressed(
    triplets_path,
    anchor=anchors, positive=positives, negative=negatives,
    anchor_idx=np.asarray(anchor_ids, dtype=np.int32),
    positive_idx=np.asarray(positive_ids, dtype=np.int32),
    negative_idx=np.asarray(negative_ids, dtype=np.int32),
)
print(f"  Saved triplets to {triplets_path}")
# ========= 2c) Save metadata for evaluation =========
triplet_meta = pd.DataFrame({
    "anchor_idx": anchor_ids,
    "positive_idx": positive_ids,
    "negative_idx": negative_ids,
    "anchor_tconst": [index_to_tconst[i] for i in anchor_ids],
    "positive_tconst": [index_to_tconst[i] for i in positive_ids],
    "negative_tconst": [index_to_tconst[i] for i in negative_ids],
})
triplet_meta.to_parquet(PROCESSED_DIR / "movie_triplets_meta.parquet", index=False)
print(f"  Saved triplet metadata to {PROCESSED_DIR / 'movie_triplets_meta.parquet'}")
print("  Triplet metadata saved for evaluation.")
# ========= 2d) Save the Annoy index for future use =========
annoy_index_path = APP_DATA_DIR / "movie_triplet_index.ann"
annoy_index.save(str(annoy_index_path))
print(f"  Saved Annoy index for triplets: {annoy_index_path}")
# ========= 2e) Save the index to tconst mapping =========
index_to_tconst_path = APP_DATA_DIR / "index_to_tconst_triplets.pkl"
pd.Series(index_to_tconst).to_pickle(index_to_tconst_path)
print(f"  Saved index to tconst mapping: {index_to_tconst_path}")
# ========= 2f) Save the master_df for reference =========
master_df_path = PROCESSED_DIR / "movie_master_df.pkl"
master_df.to_pickle(master_df_path)
print(f"  Saved master_df for reference: {master_df_path}")
# ========= 2g) Save the embeddings for reference =========
embeddings_path = PROCESSED_DIR / "movie_embeddings.npy"
np.save(embeddings_path, embeddings)
print(f"  Saved embeddings for reference: {embeddings_path}")
# ========= 2h) Save the Annoy index for content-based model =========
content_annoy_path = APP_DATA_DIR / "movie_content_index.ann"
if not content_annoy_path.exists():
    print("[align] Rebuilding Annoy index for content-based model...")
    content_annoy = AnnoyIndex(embeddings.shape[1], "angular")
    for i, vec in enumerate(embeddings):
        content_annoy.add_item(i, vec.tolist())
    content_annoy.build(50)
    content_annoy.save(str(content_annoy_path))
    print(f"[align] Saved rebuilt Annoy index for content-based model: {content_annoy_path}")
else:
    print(f"[align] Content-based Annoy index already exists: {content_annoy_path}")
# ========= 2i) Save the index to tconst mapping for content-based model =========
content_index_to_tconst_path = APP_DATA_DIR / "index_to_tconst_content.pkl"
pd.Series(index_to_tconst).to_pickle(content_index_to_tconst_path)
print(f"[align] Saved index to tconst mapping for content-based model: {content_index_to_tconst_path}")
# ========= 2j) Save the master_df for content-based model =========
content_master_df_path = PROCESSED_DIR / "movie_master_df_content.pkl"
master_df.to_pickle(content_master_df_path)
print(f"[align] Saved master_df for content-based model: {content_master_df_path}")
# ========= 2k) Save the embeddings for content-based model =========
content_embeddings_path = PROCESSED_DIR / "movie_embeddings_content.npy"
np.save(content_embeddings_path, embeddings)
print(f"[align] Saved embeddings for content-based model: {content_embeddings_path}")
# ========= 2l) Save the Annoy index for candidate model =========
candidate_annoy_path = APP_DATA_DIR / "movie_candidate_index.ann"
if not candidate_annoy_path.exists():
    print("[align] Rebuilding Annoy index for candidate model...")
    candidate_annoy = AnnoyIndex(embeddings.shape[1], "angular")
    for i, vec in enumerate(embeddings):
        candidate_annoy.add_item(i, vec.tolist())
    candidate_annoy.build(50)
    candidate_annoy.save(str(candidate_annoy_path))
    print(f"[align] Saved rebuilt Annoy index for candidate model: {candidate_annoy_path}")
else:
    print(f"[align] Candidate model Annoy index already exists: {candidate_annoy_path}")
    
# ========= 2m) Save the index to tconst mapping for candidate model =========
candidate_index_to_tconst_path = APP_DATA_DIR / "index_to_tconst_candidate.pkl"
pd.Series(index_to_tconst).to_pickle(candidate_index_to_tconst_path)
print(f"[align] Saved index to tconst mapping for candidate model: {candidate_index_to_tconst_path}")

# ========= 2n) Save the master_df for candidate model =========
candidate_master_df_path = PROCESSED_DIR / "movie_master_df_candidate.pkl"
master_df.to_pickle(candidate_master_df_path)
print(f"[align] Saved master_df for candidate model: {candidate_master_df_path}")

# ========= 3) Model =========
print("\nStep 3: Building the Two-Tower model...")

class MovieTower(tf.keras.Model):
    def __init__(self, input_dim: int, embedding_dimension: int = 64):
        super().__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,), name="tower_input"),
            tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-6)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embedding_dimension),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),
        ], name="movie_tower")

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

class TwoTowerRecommender(tfrs.Model):
    def __init__(self, input_dim: int, embedding_dimension: int = 64, margin: float = 0.2):
        super().__init__()
        self.query_tower = MovieTower(input_dim=input_dim, embedding_dimension=embedding_dimension)
        self.candidate_tower = MovieTower(input_dim=input_dim, embedding_dimension=embedding_dimension)
        self.margin = margin

    @staticmethod
    def _cosine_distance(a, b):
        sim = tf.reduce_sum(a * b, axis=1)
        return 1.0 - sim

    def train_step(self, data):
        anchor = data["anchor"]; pos = data["positive"]; neg = data["negative"]
        with tf.GradientTape() as tape:
            qe = self.query_tower(anchor, training=True)
            pe = self.candidate_tower(pos, training=True)
            ne = self.candidate_tower(neg, training=True)
            d_pos = self._cosine_distance(qe, pe)
            d_neg = self._cosine_distance(qe, ne)
            loss = tf.reduce_mean(tf.nn.relu(self.margin + d_pos - d_neg))
            loss += tf.add_n(self.losses) if self.losses else 0.0
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

input_dim = embeddings.shape[1]
model = TwoTowerRecommender(input_dim=input_dim, embedding_dimension=64, margin=0.5)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))

# ========= 4) Train =========
print("\nStep 4: Training the model...")
history = model.fit(dataset, epochs=5, verbose=1)

# ========= 5) Save =========
print("\nStep 5: Saving the trained towers and exports...")
out_dir = PROJECT_ROOT / "movie_recommender_app" / "models"
out_dir.mkdir(parents=True, exist_ok=True)

query_path = out_dir / "query_tower_model.keras"
cand_path  = out_dir / "candidate_tower_model.keras"
savedmodel_dir = out_dir / "candidate_tower_model_saved"

model.query_tower.save(str(query_path))
model.candidate_tower.save(str(cand_path))
print(f"Saved Keras models:\n  {query_path}\n  {cand_path}")

model.candidate_tower.export(str(savedmodel_dir))
print(f"Exported SavedModel to: {savedmodel_dir}")

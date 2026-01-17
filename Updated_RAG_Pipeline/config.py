"""
Configuration file for RAG system
Defines all paths, parameters, and settings
"""
import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "Updated_RAG_Pipeline/models"
EMBEDDINGS_DIR = PROJECT_ROOT / "Updated_RAG_Pipeline/embeddings"
RESULTS_DIR = PROJECT_ROOT / "Updated_RAG_Pipeline/results"
FAISS_INDEX_DIR = PROJECT_ROOT / "Updated_RAG_Pipeline/faiss_indices"

# Create directories if they don't exist
for dir_path in [DATA_DIR, MODELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, FAISS_INDEX_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Dataset configuration
DATASET_NAME = "TREC-COVID"
MAX_DOCS = None  # Set to None for full dataset, or a number for testing

# GloVe embeddings
GLOVE_PATH = DATA_DIR / "glove.6B" / "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
GLOVE_DIM = 300

# LDA configuration
N_TOPICS = 100  # Number of topics for LDA
N_TOP_WORDS = 20  # Keep top N words per topic for sparse representation
LDA_ITERATIONS = 1500
LDA_RANDOM_STATE = 42

# LOTT configuration
GAUSSIAN_FWHM = 25  # Full-width half-maximum for Gaussian reference
P_NORM = 1  # Use W1 distance (p=1)

# BERT configuration
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Fast and efficient SBERT model
BERT_BATCH_SIZE = 32

# Retrieval configuration
TOP_K_VALUES = [1, 5, 10, 20]  # K values for evaluation metrics
N_NEIGHBORS_KNN = 7  # For KNN classification if needed

# File paths for saved artifacts
LDA_MODEL_PATH = MODELS_DIR / "lda_model.pkl"
LDA_TOPICS_PATH = MODELS_DIR / "lda_topics.npy"
LDA_CENTERS_PATH = MODELS_DIR / "lda_centers.npy"
TOPIC_COST_MATRIX_PATH = MODELS_DIR / "topic_cost_matrix.npy"
VOCAB_PATH = MODELS_DIR / "vocabulary.pkl"
WORD_EMBEDDINGS_PATH = MODELS_DIR / "word_embeddings.npy"

# Embedding paths
BERT_EMBEDDINGS = EMBEDDINGS_DIR / "bert_embeddings.pkl"
LOTT_EMBEDDINGS = EMBEDDINGS_DIR / "lott_embeddings.npy"
TOPIC_PROPORTIONS = EMBEDDINGS_DIR / "topic_proportions.npy"

# FAISS index paths
BERT_FAISS_INDEX = FAISS_INDEX_DIR / "bert_index.faiss"
LOTT_FAISS_INDEX = FAISS_INDEX_DIR / "lott_index.faiss"

# Document store paths
DOCUMENTS_PATH = DATA_DIR / "documents.pkl"  # Single corpus, no split
QUERIES_PATH = DATA_DIR / "queries.pkl"
QRELS_PATH = DATA_DIR / "qrels.pkl"

# Results paths
METRICS_RESULTS_PATH = RESULTS_DIR / "evaluation_metrics.json"
TIMING_RESULTS_PATH = RESULTS_DIR / "timing_results.json"
COMPARISON_PLOT_PATH = RESULTS_DIR / "comparison_plot.png"

# Random seed for reproducibility
RANDOM_SEED = 42

# Display settings
VERBOSE = True

def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("RAG SYSTEM CONFIGURATION")
    print("=" * 80)
    print(f"Dataset: {DATASET_NAME}")
    print(f"Number of Topics: {N_TOPICS}")
    print(f"BERT Model: {BERT_MODEL}")
    print(f"GloVe Embeddings: {GLOVE_PATH}")
    print(f"Top-K Values: {TOP_K_VALUES}")
    print("=" * 80)
    print()

if __name__ == "__main__":
    print_config()
    print("All directories created successfully!")
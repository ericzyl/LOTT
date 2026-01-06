"""
Configuration file for RAG system
Defines all paths, parameters, and settings
"""
import os
from pathlib import Path

# Base Dirs
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "RAG_BeIR_Pipeline/models"
EMBEDDINGS_DIR = PROJECT_ROOT / "RAG_BeIR_Pipeline/embeddings"
RESULTS_DIR = PROJECT_ROOT / "RAG_BeIR_Pipeline/results"
FAISS_INDEX_DIR = PROJECT_ROOT / "RAG_BeIR_Pipeline/faiss_indices"

for dir_path in [DATA_DIR, MODELS_DIR, EMBEDDINGS_DIR, RESULTS_DIR, FAISS_INDEX_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Dataset Config
DATASET_NAME = "BeIR/trec-covid"
# DATASET_SUBSET = "generated-queries"
MAX_DOCS = None  # None for Full Dataset, could be set to a Number during Testing

# GLOVE_PATH = DATA_DIR / "glove.6B" / "glove.6B.300d.txt"
GLOVE_PATH = "/kaggle/input/bert-lott-exps-datasets/glove.6B.300d.txt"
GLOVE_DIM = 300

# LDA Config
N_TOPICS = 100  # Number of Topics for LDA
N_TOP_WORDS = 20  # Keeping top N words per Topic for Sparse representation
LDA_ITERATIONS = 1500
LDA_RANDOM_STATE = 42

# LOTT Config
GAUSSIAN_FWHM = 25  # Full-width half-maximum for Gaussian reference
P_NORM = 1  # Using W1 Distance (p=1)

# BERT Config
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_BATCH_SIZE = 32

# Retrieval Config
TOP_K_VALUES = [1, 5, 10, 20]  # K values for Evaluation Metrics
N_NEIGHBORS_KNN = 7  # For KNN Classification

# File Paths for saved Artifacts
LDA_MODEL_PATH = MODELS_DIR / "lda_model.pkl"
LDA_TOPICS_PATH = MODELS_DIR / "lda_topics.npy"
LDA_CENTERS_PATH = MODELS_DIR / "lda_centers.npy"
TOPIC_COST_MATRIX_PATH = MODELS_DIR / "topic_cost_matrix.npy"
VOCAB_PATH = MODELS_DIR / "vocabulary.pkl"
WORD_EMBEDDINGS_PATH = MODELS_DIR / "word_embeddings.npy"

# Embedding Paths
BERT_TRAIN_EMBEDDINGS = EMBEDDINGS_DIR / "bert_train_embeddings.npy"
BERT_TEST_EMBEDDINGS = EMBEDDINGS_DIR / "bert_test_embeddings.npy"
LOTT_TRAIN_EMBEDDINGS = EMBEDDINGS_DIR / "lott_train_embeddings.npy"
LOTT_TEST_EMBEDDINGS = EMBEDDINGS_DIR / "lott_test_embeddings.npy"
TOPIC_PROPORTIONS_TRAIN = EMBEDDINGS_DIR / "topic_proportions_train.npy"
TOPIC_PROPORTIONS_TEST = EMBEDDINGS_DIR / "topic_proportions_test.npy"

# FAISS Index Paths
BERT_FAISS_INDEX = FAISS_INDEX_DIR / "bert_index.faiss"
LOTT_FAISS_INDEX = FAISS_INDEX_DIR / "lott_index.faiss"

# Document Store Paths
DOCUMENTS_TRAIN_PATH = DATA_DIR / "documents_train.pkl"
DOCUMENTS_TEST_PATH = DATA_DIR / "documents_test.pkl"
QUERIES_PATH = DATA_DIR / "queries.pkl"
QRELS_PATH = DATA_DIR / "qrels.pkl"

# Results Paths
METRICS_RESULTS_PATH = RESULTS_DIR / "evaluation_metrics.json"
TIMING_RESULTS_PATH = RESULTS_DIR / "timing_results.json"
COMPARISON_PLOT_PATH = RESULTS_DIR / "comparison_plot.png"

# Random Seed for Reproducibility
RANDOM_SEED = 42

# Display Settings
VERBOSE = True

def print_config():
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
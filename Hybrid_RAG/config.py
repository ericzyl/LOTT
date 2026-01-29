from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
for dir_path in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# Dataset selection
AVAILABLE_DATASETS = ["trec-covid", "msmarco"]

# GloVe embeddings path
GLOVE_PATH = DATA_DIR / "wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
GLOVE_DIM = 300

# BERT configuration
BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_BATCH_SIZE = 32
BERT_DIM = 384

# LDA configuration
N_TOPICS = 100
N_TOP_WORDS = 20  # Sparse topics
LDA_MAX_ITER = 1500
LDA_RANDOM_STATE = 42

# Retrieval configuration
K_RETRIEVAL = 100  # Retrieve top-100 with BERT
TOP_K_VALUES = [1, 5, 10, 20]  # Final evaluation at these K values

# Preprocessing
MIN_WORD_LENGTH = 3
MAX_VOCAB_SIZE = 10000

# Cache paths (per dataset)
def get_cache_paths(dataset_name):
    """Get cache file paths for a specific dataset"""
    cache_subdir = CACHE_DIR / dataset_name
    cache_subdir.mkdir(exist_ok=True, parents=True)
    
    return {
        'corpus': cache_subdir / 'corpus.pkl',
        'queries': cache_subdir / 'queries.pkl',
        'qrels': cache_subdir / 'qrels.pkl',
        'vocab': cache_subdir / 'vocabulary.pkl',
        'word_embeddings': cache_subdir / 'word_embeddings.npy',
        'bow_data': cache_subdir / 'bow_data.npy',
        'doc_ids': cache_subdir / 'doc_ids.pkl',
        'lda_model': cache_subdir / 'lda_model.pkl',
        'lda_topics': cache_subdir / 'lda_topics.npy',
        'lda_centers': cache_subdir / 'lda_centers.npy',
        'topic_cost_matrix': cache_subdir / 'topic_cost_matrix.npy',
        'topic_proportions': cache_subdir / 'topic_proportions.npy',
        'query_topic_proportions': cache_subdir / 'query_topic_proportions.npy',
        'bert_embeddings': cache_subdir / 'bert_doc_embeddings.npy',
        'bert_query_embeddings': cache_subdir / 'bert_query_embeddings.npy',
        'lott_embeddings': cache_subdir / 'lott_doc_embeddings.npy',
        'lott_query_embeddings': cache_subdir / 'lott_query_embeddings.npy',
        'faiss_index': cache_subdir / 'bert_faiss.index'
    }

# Results paths
def get_results_path(dataset_name):
    """Get results file path for a dataset"""
    return RESULTS_DIR / f"{dataset_name}_results.json"

def get_plot_path(dataset_name):
    """Get plot file path for a dataset"""
    return RESULTS_DIR / f"{dataset_name}_comparison.png"

# Display
VERBOSE = True

def print_config(dataset_name):
    """Print configuration"""
    print("="*80)
    print(f"HYBRID RAG CONFIGURATION - {dataset_name.upper()}")
    print("="*80)
    print(f"BERT Model: {BERT_MODEL}")
    print(f"LDA Topics: {N_TOPICS}")
    print(f"Retrieval: Top-{K_RETRIEVAL}")
    print(f"Evaluation: @{TOP_K_VALUES}")
    print(f"Cache directory: {CACHE_DIR / dataset_name}")
    print("="*80)
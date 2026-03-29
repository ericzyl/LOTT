from pathlib import Path

# PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = "/content/drive/MyDrive/LOTT"
# DATA_DIR = PROJECT_ROOT / "data"
# CACHE_DIR = PROJECT_ROOT / "LOTT_then_BERT_Hybrid/cache"
# RESULTS_DIR = PROJECT_ROOT / "LOTT_then_BERT_Hybrid/results"
DATA_DIR = PROJECT_ROOT
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

for dir_path in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

AVAILABLE_DATASETS = ["msmarco"]

GLOVE_PATH = DATA_DIR / "dolma_300_2024_1.2M.100_combined.txt"
GLOVE_DIM = 300

BERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BERT_BATCH_SIZE = 32
BERT_DIM = 384

N_TOPICS = 100
N_TOP_WORDS = 20
LDA_MAX_ITER = 1000
LDA_RANDOM_STATE = 42

# LOTT retrieves this many candidates first (broad initial retrieval)
K_LOTT_RETRIEVAL = 200

# BERT then reranks down to this final count
K_FINAL = 10

# Evaluation K values
TOP_K_VALUES = [1, 5, 10, 20]

MIN_WORD_LENGTH = 3
MAX_VOCAB_SIZE = 10000

GAUSSIAN_FWHM = 25
P_NORM = 1

MAX_DOCS = None  # Set to an integer (e.g. 100000) to cap corpus size for testing


def get_cache_paths(dataset_name: str) -> dict:
    cache_subdir = CACHE_DIR / dataset_name
    cache_subdir.mkdir(exist_ok=True, parents=True)
    return {
        'corpus':                   cache_subdir / 'corpus.pkl',
        'queries':                  cache_subdir / 'queries.pkl',
        'qrels':                    cache_subdir / 'qrels.pkl',
        'vocab':                    cache_subdir / 'vocabulary.pkl',
        'word_embeddings':          cache_subdir / 'word_embeddings.npy',
        'bow_data':                 cache_subdir / 'bow_data.npy',
        'doc_ids':                  cache_subdir / 'doc_ids.pkl',
        'lda_model':                cache_subdir / 'lda_model.pkl',
        'lda_topics':               cache_subdir / 'lda_topics.npy',
        'lda_centers':              cache_subdir / 'lda_centers.npy',
        'topic_cost_matrix':        cache_subdir / 'topic_cost_matrix.npy',
        'topic_proportions':        cache_subdir / 'topic_proportions.npy',
        'query_topic_proportions':  cache_subdir / 'query_topic_proportions.npy',
        'bert_embeddings':          cache_subdir / 'bert_doc_embeddings.npy',
        'bert_query_embeddings':    cache_subdir / 'bert_query_embeddings.npy',
        'lott_embeddings':          cache_subdir / 'lott_doc_embeddings.npy',
        'lott_query_embeddings':    cache_subdir / 'lott_query_embeddings.npy',
        'faiss_lott_index':         cache_subdir / 'lott_faiss.index',
    }


def get_results_path(dataset_name: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_results.json"


def get_plot_path(dataset_name: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_comparison.png"


def get_timing_plot_path(dataset_name: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_timing.png"


VERBOSE = True


def print_config(dataset_name: str):
    print("=" * 80)
    print(f"LOTT → BERT HYBRID RAG  |  dataset: {dataset_name.upper()}")
    print("=" * 80)
    print(f"  BERT model          : {BERT_MODEL}")
    print(f"  LDA topics          : {N_TOPICS}")
    print(f"  LOTT initial pool   : top-{K_LOTT_RETRIEVAL}")
    print(f"  BERT final results  : top-{K_FINAL}")
    print(f"  Evaluation K values : {TOP_K_VALUES}")
    print(f"  Cache directory     : {CACHE_DIR / dataset_name}")
    print("=" * 80)
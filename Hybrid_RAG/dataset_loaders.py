import pickle
from typing import Dict, Tuple
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import config


def load_trec_covid() -> Tuple[Dict, Dict, Dict]:
    print("Loading TREC-COVID dataset...")
    
    # Check cache first
    cache = config.get_cache_paths('trec-covid')
    if cache['corpus'].exists() and cache['queries'].exists():
        print("Loading from cache...")
        with open(cache['corpus'], 'rb') as f:
            corpus = pickle.load(f)
        with open(cache['queries'], 'rb') as f:
            queries = pickle.load(f)
        with open(cache['qrels'], 'rb') as f:
            qrels = pickle.load(f)
        return corpus, queries, qrels
    
    # Download and load
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip"
    data_path = util.download_and_unzip(url, str(config.DATA_DIR))
    beir_corpus, beir_queries, beir_qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    # Process corpus
    corpus = {}
    print(f"Processing {len(beir_corpus)} documents...")
    for doc_id, doc_data in beir_corpus.items():
        title = doc_data.get('title', '')
        text = doc_data.get('text', '')
        full_text = f"{title}. {text}" if title else text
        corpus[doc_id] = {
            'text': full_text,
            'title': title,
            'doc_id': doc_id
        }
    
    queries = beir_queries
    qrels = beir_qrels
    
    # Cache
    with open(cache['corpus'], 'wb') as f:
        pickle.dump(corpus, f)
    with open(cache['queries'], 'wb') as f:
        pickle.dump(queries, f)
    with open(cache['qrels'], 'wb') as f:
        pickle.dump(qrels, f)
    
    print(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def load_msmarco() -> Tuple[Dict, Dict, Dict]:
    print("Loading MS-MARCO dataset...")
    
    # Check cache
    cache = config.get_cache_paths('msmarco')
    if cache['corpus'].exists() and cache['queries'].exists():
        print("Loading from cache...")
        with open(cache['corpus'], 'rb') as f:
            corpus = pickle.load(f)
        with open(cache['queries'], 'rb') as f:
            queries = pickle.load(f)
        with open(cache['qrels'], 'rb') as f:
            qrels = pickle.load(f)
        return corpus, queries, qrels
    
    # Download and load
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip"
    data_path = util.download_and_unzip(url, str(config.DATA_DIR))
    beir_corpus, beir_queries, beir_qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
    
    # Process corpus
    corpus = {}
    print(f"Processing {len(beir_corpus)} documents...")
    for idx, (doc_id, doc_data) in enumerate(beir_corpus.items()):
        if idx % 100000 == 0:
            print(f"  Processed {idx}/{len(beir_corpus)} documents")
        
        title = doc_data.get('title', '')
        text = doc_data.get('text', '')
        full_text = f"{title}. {text}" if title else text
        corpus[doc_id] = {
            'text': full_text,
            'title': title,
            'doc_id': doc_id
        }
    
    queries = beir_queries
    qrels = beir_qrels
    
    # Cache
    print("Caching processed data...")
    with open(cache['corpus'], 'wb') as f:
        pickle.dump(corpus, f)
    with open(cache['queries'], 'wb') as f:
        pickle.dump(queries, f)
    with open(cache['qrels'], 'wb') as f:
        pickle.dump(qrels, f)
    
    print(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    return corpus, queries, qrels


def load_dataset(dataset_name: str) -> Tuple[Dict, Dict, Dict]:
    if dataset_name == "trec-covid":
        return load_trec_covid()
    elif dataset_name == "msmarco":
        return load_msmarco()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {config.AVAILABLE_DATASETS}")


if __name__ == "__main__":
    # Test loading
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "trec-covid"
    corpus, queries, qrels = load_dataset(dataset)
    print(f"\nDataset: {dataset}")
    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Qrels: {len(qrels)}")
"""
BERT-based retrieval with FAISS
"""
import numpy as np
import faiss
import pickle
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer
import config


class BERTRetriever:
    """BERT embedding + FAISS retrieval"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.cache = config.get_cache_paths(dataset_name)
        self.model = None
        self.index = None
        self.doc_ids = None
        self.doc_embeddings = None
        
    def build_index(self, corpus: Dict, doc_ids: List[str]):
        """Build BERT embeddings and FAISS index"""
        
        # Load model
        print(f"Loading BERT model: {config.BERT_MODEL}...")
        self.model = SentenceTransformer(config.BERT_MODEL)
        self.doc_ids = doc_ids
        
        # Check for cached embeddings
        if self.cache['bert_embeddings'].exists():
            print("Loading BERT embeddings from cache...")
            self.doc_embeddings = np.load(self.cache['bert_embeddings'])
        else:
            # Generate embeddings
            print(f"Generating BERT embeddings for {len(doc_ids)} documents...")
            texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
            self.doc_embeddings = self.model.encode(
                texts,
                batch_size=config.BERT_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Cache
            np.save(self.cache['bert_embeddings'], self.doc_embeddings)
            print(f"Cached embeddings to {self.cache['bert_embeddings']}")
        
        # Build FAISS index
        if self.cache['faiss_index'].exists():
            print("Loading FAISS index from cache...")
            self.index = faiss.read_index(str(self.cache['faiss_index']))
        else:
            print("Building FAISS index...")
            embeddings_normalized = self.doc_embeddings.astype('float32')
            faiss.normalize_L2(embeddings_normalized)
            
            self.index = faiss.IndexFlatIP(config.BERT_DIM)
            self.index.add(embeddings_normalized)
            
            # Cache index
            faiss.write_index(self.index, str(self.cache['faiss_index']))
            print(f"Cached index to {self.cache['faiss_index']}")
        
        print(f"BERT retriever ready! {self.index.ntotal} documents indexed.")
    
    def encode_queries(self, queries: Dict) -> Tuple[np.ndarray, List[str]]:
        """Encode queries with BERT"""
        
        query_ids = list(queries.keys())
        
        # Check cache
        if self.cache['bert_query_embeddings'].exists():
            print("Loading query embeddings from cache...")
            query_embeddings = np.load(self.cache['bert_query_embeddings'])
            return query_embeddings, query_ids
        
        # Generate
        print(f"Encoding {len(queries)} queries with BERT...")
        query_texts = [queries[qid] for qid in query_ids]
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=config.BERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Cache
        np.save(self.cache['bert_query_embeddings'], query_embeddings)
        
        return query_embeddings, query_ids
    
    def retrieve(self, query_embeddings: np.ndarray, k: int = 100) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Retrieve top-k documents for each query
        
        Returns:
            retrieved_doc_ids: List of lists of doc IDs
            scores: List of lists of similarity scores
        """
        print(f"Retrieving top-{k} documents for {len(query_embeddings)} queries...")
        
        # Normalize queries
        queries_normalized = query_embeddings.astype('float32')
        faiss.normalize_L2(queries_normalized)
        
        # Search
        similarities, indices = self.index.search(queries_normalized, k)
        
        # Convert to doc IDs
        retrieved_doc_ids = []
        all_scores = []
        
        for i in range(len(query_embeddings)):
            doc_ids = [self.doc_ids[idx] for idx in indices[i]]
            scores = similarities[i].tolist()
            retrieved_doc_ids.append(doc_ids)
            all_scores.append(scores)
        
        return retrieved_doc_ids, all_scores


if __name__ == "__main__":
    import sys
    from dataset_loaders import load_dataset
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else "trec-covid"
    
    # Load data
    corpus, queries, _ = load_dataset(dataset)
    doc_ids = list(corpus.keys())
    
    # Build retriever
    retriever = BERTRetriever(dataset)
    retriever.build_index(corpus, doc_ids)
    
    # Test retrieval
    query_embs, query_ids = retriever.encode_queries(queries)
    retrieved, scores = retriever.retrieve(query_embs, k=10)
    
    print(f"\nTest: Retrieved {len(retrieved[0])} docs for first query")
    print(f"First doc ID: {retrieved[0][0]}, Score: {scores[0][0]:.4f}")
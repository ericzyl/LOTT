"""
BERT-based RAG system using FAISS for efficient retrieval
"""
import numpy as np
import faiss
import pickle
import time
from typing import Dict, List, Tuple
import config


class BERTRAGSystem:
    """RAG system using BERT embeddings and FAISS index"""
    
    def __init__(self, embeddings: np.ndarray, doc_ids: List[str], 
                 corpus: Dict[str, Dict]):
        self.embeddings = embeddings
        self.doc_ids = doc_ids
        self.corpus = corpus
        self.index = None
        self.embedding_dim = embeddings.shape[1]
        
    def build_index(self):
        """Build FAISS index for efficient similarity search"""
        print(f"Building FAISS index with {len(self.embeddings)} vectors...")
        start_time = time.time()
        
        # Normalize embeddings for cosine similarity
        embeddings_normalized = self.embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # Create FAISS index (Inner Product = Cosine Similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_normalized)
        
        build_time = time.time() - start_time
        print(f"FAISS index built in {build_time:.2f} seconds")
        print(f"Index contains {self.index.ntotal} vectors")
        
        return build_time
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        """
        Search for top-k most similar documents
        
        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
        
        Returns:
            Tuple of (doc_ids, similarity_scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query
        query_normalized = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Search
        similarities, indices = self.index.search(query_normalized, k)
        
        # Get document IDs
        doc_ids = [self.doc_ids[idx] for idx in indices[0]]
        scores = similarities[0].tolist()
        
        return doc_ids, scores
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10, 
                     show_progress: bool = True) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Array of query embeddings (n_queries, embedding_dim)
            k: Number of results per query
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (list of doc_id lists, list of score lists)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        all_doc_ids = []
        all_scores = []
        
        if show_progress:
            print(f"Searching {len(query_embeddings)} queries...")
        
        for i in range(len(query_embeddings)):
            if show_progress and i % 1000 == 0:
                print(f"  Progress: {i}/{len(query_embeddings)} queries")
            doc_ids, scores = self.search(query_embeddings[i], k=k)
            all_doc_ids.append(doc_ids)
            all_scores.append(scores)
        
        return all_doc_ids, all_scores
    
    def save_index(self, path=None):
        """Save FAISS index to disk"""
        if path is None:
            path = config.BERT_FAISS_INDEX
        
        faiss.write_index(self.index, str(path))
        print(f"FAISS index saved to {path}")
    
    def load_index(self, path=None):
        """Load FAISS index from disk"""
        if path is None:
            path = config.BERT_FAISS_INDEX
        
        self.index = faiss.read_index(str(path))
        print(f"FAISS index loaded from {path}")
        print(f"Index contains {self.index.ntotal} vectors")


def build_bert_rag_system():
    """Build and save BERT RAG system"""
    print("\n" + "="*80)
    print("BUILDING BERT RAG SYSTEM")
    print("="*80)
    
    # Load embeddings
    print("Loading BERT embeddings...")
    embeddings = np.load(config.BERT_EMBEDDINGS)
    
    # Load document IDs
    print("Loading document metadata...")
    with open(config.SAVE_DATA_DIR / 'pipeline_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    doc_ids = metadata['doc_ids']
    
    # Load corpus
    print("Loading corpus...")
    with open(config.DOCUMENTS_PATH, 'rb') as f:
        corpus = pickle.load(f)
    
    # Initialize RAG system
    rag_system = BERTRAGSystem(embeddings, doc_ids, corpus)
    
    # Build index
    build_time = rag_system.build_index()
    
    # Save index
    rag_system.save_index()
    
    print("\n" + "="*80)
    print("BERT RAG SYSTEM READY")
    print("="*80)
    print(f"Documents indexed: {len(doc_ids)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index build time: {build_time:.2f} seconds")
    
    return rag_system


def test_bert_rag():
    """Test BERT RAG system with sample queries"""
    print("\n" + "="*80)
    print("TESTING BERT RAG SYSTEM")
    print("="*80)
    
    # Load system components
    rag_system = build_bert_rag_system()
    
    # Load test queries
    with open(config.QUERIES_PATH, 'rb') as f:
        queries = pickle.load(f)
    
    # Generate query embeddings for a few test queries
    from bert_embeddings import BERTEmbeddingGenerator
    
    test_query_ids = list(queries.keys())[:5]  # Test with first 5 queries
    test_queries = {qid: queries[qid] for qid in test_query_ids}
    
    generator = BERTEmbeddingGenerator()
    query_embeddings, query_ids = generator.generate_query_embeddings(test_queries)
    
    # Perform retrieval
    print("\nPerforming retrieval for test queries...")
    start_time = time.time()
    doc_ids_list, scores_list = rag_system.batch_search(query_embeddings, k=5, show_progress=True)
    retrieval_time = time.time() - start_time
    
    print(f"\nRetrieval completed in {retrieval_time:.2f} seconds")
    print(f"Average time per query: {retrieval_time / len(test_queries):.4f} seconds")
    
    # Display results
    print("\n" + "="*80)
    print("SAMPLE RETRIEVAL RESULTS")
    print("="*80)
    
    for i, qid in enumerate(query_ids[:3]):  # Show first 3
        print(f"\nQuery {i+1}: {test_queries[qid]}")
        print(f"Top 3 retrieved documents:")
        
        for j, (doc_id, score) in enumerate(zip(doc_ids_list[i][:3], scores_list[i][:3])):
            doc_text = rag_system.corpus[doc_id]['text'][:200]  # First 200 chars
            print(f"  {j+1}. (Score: {score:.4f})")
            print(f"     {doc_text}...")
            print()


if __name__ == "__main__":
    # Build the system
    rag_system = build_bert_rag_system()
    
    # Run tests
    print("\nWould you like to test the system? (This will generate query embeddings)")
    # Uncomment to run tests:
    # test_bert_rag()
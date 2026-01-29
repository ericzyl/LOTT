# LOTT-based RAG system using FAISS for efficient retrieval
import numpy as np
import faiss
import pickle
import time
from typing import Dict, List, Tuple
# from tqdm import tqdm
import config


class LOTTRAGSystem:
    def __init__(self, train_embeddings: np.ndarray, train_doc_ids: List[str],
                 train_corpus: Dict[str, Dict]):
        self.train_embeddings = train_embeddings
        self.train_doc_ids = train_doc_ids
        self.train_corpus = train_corpus
        self.index = None
        self.embedding_dim = train_embeddings.shape[1]
        
    def build_index(self):
        # Building FAISS index for efficient similarity search
        print(f"Building FAISS index with {len(self.train_embeddings)} vectors...")
        start_time = time.time()
        
        # Normalizing embeddings for cosine similarity
        embeddings_normalized = self.train_embeddings.astype('float32')
        faiss.normalize_L2(embeddings_normalized)
        
        # Creating FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings_normalized)
        
        build_time = time.time() - start_time
        print(f"FAISS index built in {build_time:.2f} seconds")
        print(f"Index contains {self.index.ntotal} vectors")
        
        return build_time
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> Tuple[List[str], List[float]]:
        # Searching for top-k most similar documents
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalizing query
        query_normalized = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_normalized)
        
        # Searching
        similarities, indices = self.index.search(query_normalized, k)
        
        # Getting document IDs
        doc_ids = [self.train_doc_ids[idx] for idx in indices[0]]
        scores = similarities[0].tolist()
        
        return doc_ids, scores
    
    def batch_search(self, query_embeddings: np.ndarray, k: int = 10,
                     show_progress: bool = True) -> Tuple[List[List[str]], List[List[float]]]:
        # Batch search for multiple queries     
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        all_doc_ids = []
        all_scores = []
        
        # iterator = tqdm(range(len(query_embeddings)), desc="Searching queries") \
        #            if show_progress else range(len(query_embeddings))
        
        # for i in iterator:
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
        if path is None:
            path = config.LOTT_FAISS_INDEX
        
        faiss.write_index(self.index, str(path))
        print(f"FAISS index saved to {path}")
    
    def load_index(self, path=None):
        if path is None:
            path = config.LOTT_FAISS_INDEX
        
        self.index = faiss.read_index(str(path))
        print(f"FAISS index loaded from {path}")
        print(f"Index contains {self.index.ntotal} vectors")


def build_lott_rag_system():
    # Building and saving LOTT RAG system
    print("\n" + "="*80)
    print("BUILDING LOTT RAG SYSTEM")
    print("="*80)
    
    print("Loading LOTT embeddings...")
    train_embeddings = np.load(config.LOTT_TRAIN_EMBEDDINGS)
    
    print("Loading document metadata...")
    with open(config.DATA_DIR / 'pipeline_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    train_doc_ids = metadata['train_doc_ids']
    
    # Loading corpus
    print("Loading corpus...")
    with open(config.DOCUMENTS_TRAIN_PATH, 'rb') as f:
        train_corpus = pickle.load(f)
    
    # Initializing RAG system
    rag_system = LOTTRAGSystem(train_embeddings, train_doc_ids, train_corpus)
    
    # Building index
    build_time = rag_system.build_index()
    
    # Saving index
    rag_system.save_index()
    
    print("\n" + "="*80)
    print("LOTT RAG SYSTEM READY")
    print("="*80)
    print(f"Documents indexed: {len(train_doc_ids)}")
    print(f"Embedding dimension: {train_embeddings.shape[1]}")
    print(f"Index build time: {build_time:.2f} seconds")
    
    return rag_system


def generate_lott_query_embeddings(queries: Dict[str, str]) -> Tuple[np.ndarray, List[str]]:
    # Generating LOTT embeddings for queries
    from preprocessing import TextPreprocessor
    from lda_trainer import LDATrainer
    from lott_embeddings import LOTTEmbeddingGenerator
    
    print("Generating LOTT embeddings for queries...")
    
    # Loading preprocessor
    preprocessor = TextPreprocessor()
    preprocessor.load_vocabulary()
    
    # Converting queries to BoW
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    
    query_bow = np.array([preprocessor.text_to_bow(text) for text in query_texts])
    
    # Loading LDA artifacts
    lda_model, topics, lda_centers, topic_cost_matrix = LDATrainer.load_artifacts()
    
    # Generating LOTT embeddings
    generator = LOTTEmbeddingGenerator(
        lda_model=lda_model,
        lda_centers=lda_centers,
        topic_cost_matrix=topic_cost_matrix,
        gaussian_fwhm=config.GAUSSIAN_FWHM
    )
    
    query_embeddings, query_topics = generator.generate_from_bow(query_bow, show_progress=True)

    # print("Compressing query embeddings with PCA...")
    # pca_path = config.MODELS_DIR / "lott_pca.pkl"
    # with open(pca_path, 'rb') as f:
    #     pca = pickle.load(f)
    
    # query_embeddings_compressed = pca.transform(query_embeddings)
    # print(f"Query embeddings compressed: {query_embeddings_compressed.shape}")
    
    # return query_embeddings_compressed, query_ids

    return query_embeddings, query_ids


def test_lott_rag():
    # Testing LOTT RAG system with sample queries
    print("\n" + "="*80)
    print("TESTING LOTT RAG SYSTEM")
    print("="*80)
    
    # Loading system components
    rag_system = build_lott_rag_system()
    
    # Loading test queries
    with open(config.QUERIES_PATH, 'rb') as f:
        queries = pickle.load(f)
    
    # Generating query embeddings for a few test queries
    test_query_ids = list(queries.keys())[:5]  # Test with first 5 queries
    test_queries = {qid: queries[qid] for qid in test_query_ids}
    
    query_embeddings, query_ids = generate_lott_query_embeddings(test_queries)
    
    # Performing retrieval
    print("\nPerforming retrieval for test queries...")
    start_time = time.time()
    doc_ids_list, scores_list = rag_system.batch_search(query_embeddings, k=5, show_progress=True)
    retrieval_time = time.time() - start_time
    
    print(f"\nRetrieval completed in {retrieval_time:.2f} seconds")
    print(f"Average time per query: {retrieval_time / len(test_queries):.4f} seconds")
    
    print("\n" + "="*80)
    print("SAMPLE RETRIEVAL RESULTS")
    print("="*80)
    
    for i, qid in enumerate(query_ids[:3]):  # Show first 3
        print(f"\nQuery {i+1}: {test_queries[qid]}")
        print(f"Top 3 retrieved documents:")
        
        for j, (doc_id, score) in enumerate(zip(doc_ids_list[i][:3], scores_list[i][:3])):
            doc_text = rag_system.train_corpus[doc_id]['text'][:200]  # First 200 chars
            print(f"  {j+1}. (Score: {score:.4f})")
            print(f"     {doc_text}...")
            print()


if __name__ == "__main__":
    rag_system = build_lott_rag_system()
    
    # Running tests
    print("\nWould you like to test the system? (This will generate query embeddings)")
    # Uncomment to run tests:
    # test_lott_rag()
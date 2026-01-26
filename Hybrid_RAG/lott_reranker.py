"""
LOTT-based reranking
Two methods: Embedding-based and OT-based
"""
import numpy as np
import ot
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import config


def makeGaussian1D(size, fwhm=25, center=None):
    """Generate 1D Gaussian distribution"""
    center = center if center is not None else size // 2
    x = np.arange(size)
    gaussian = np.exp(-4 * np.log(2) * ((x - center)**2) / fwhm**2)
    return gaussian / gaussian.sum()


def lot_embedding(coupling_matrix, lda_centers):
    """Compute LOT embedding from OT coupling"""
    T_mu_sigma = []
    
    for i in range(coupling_matrix.shape[0]):
        source_weight = coupling_matrix[i].sum()
        
        if source_weight == 0:
            y_coor = np.zeros(lda_centers.shape[1], dtype=float)
        else:
            nonzero_indices = np.where(coupling_matrix[i] > 0)[0]
            weights = coupling_matrix[i][nonzero_indices]
            positions = lda_centers[nonzero_indices]
            y_coor = (weights[:, None] * positions).sum(axis=0) / source_weight
        
        T_mu_sigma.append(y_coor)
    
    return np.array(T_mu_sigma).flatten()


class LOTTReranker:
    """LOTT-based reranking with two methods"""
    
    def __init__(self, dataset_name: str, lda_module):
        self.dataset_name = dataset_name
        self.cache = config.get_cache_paths(dataset_name)
        self.lda_module = lda_module
        self.gaussian_ref = makeGaussian1D(config.N_TOPICS, fwhm=25)
        
        self.doc_topic_props = None
        self.query_topic_props = None
        self.doc_lott_embs = None
        self.query_lott_embs = None
        
    def prepare_documents(self, bow_data: np.ndarray):
        """Prepare document topic proportions and LOTT embeddings"""
        
        # Infer topic proportions
        self.doc_topic_props = self.lda_module.infer_topics(bow_data, 'topic_proportions')
        
        # Check for cached LOTT embeddings
        if self.cache['lott_embeddings'].exists():
            print("Loading LOTT doc embeddings from cache...")
            self.doc_lott_embs = np.load(self.cache['lott_embeddings'])
        else:
            # Generate LOTT embeddings
            print(f"Generating LOTT embeddings for {len(bow_data)} documents...")
            self.doc_lott_embs = self._create_lott_embeddings(self.doc_topic_props)
            
            # Cache
            np.save(self.cache['lott_embeddings'], self.doc_lott_embs)
        
        print(f"Document topic props: {self.doc_topic_props.shape}")
        print(f"Document LOTT embeddings: {self.doc_lott_embs.shape}")
    
    def prepare_queries(self, queries: Dict, corpus: Dict, vocab_data: dict):
        """Prepare query topic proportions and LOTT embeddings"""
        
        # Need to convert queries to BoW first
        from preprocessing import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        preprocessor.vocab = vocab_data['vocab']
        preprocessor.word_to_idx = vocab_data['word_to_idx']
        
        # Convert queries to BoW
        query_ids = list(queries.keys())
        query_bow = np.array([
            preprocessor.text_to_bow(queries[qid]) 
            for qid in query_ids
        ])
        
        # Infer topic proportions
        self.query_topic_props = self.lda_module.infer_topics(query_bow, 'query_topic_proportions')
        
        # Check for cached LOTT embeddings
        if self.cache['lott_query_embeddings'].exists():
            print("Loading LOTT query embeddings from cache...")
            self.query_lott_embs = np.load(self.cache['lott_query_embeddings'])
        else:
            # Generate LOTT embeddings
            print(f"Generating LOTT embeddings for {len(queries)} queries...")
            self.query_lott_embs = self._create_lott_embeddings(self.query_topic_props)
            
            # Cache
            np.save(self.cache['lott_query_embeddings'], self.query_lott_embs)
        
        print(f"Query topic props: {self.query_topic_props.shape}")
        print(f"Query LOTT embeddings: {self.query_lott_embs.shape}")
        
        return query_ids
    
    def _create_lott_embeddings(self, topic_proportions: np.ndarray) -> np.ndarray:
        """Create LOTT embeddings from topic proportions"""
        lot_embeddings = []
        
        for i in range(len(topic_proportions)):
            if i % 1000 == 0:
                print(f"  Progress: {i}/{len(topic_proportions)}")
            
            doc_topics = topic_proportions[i].reshape(-1)
            if doc_topics.sum() > 0:
                doc_topics = doc_topics / doc_topics.sum()
            else:
                doc_topics = np.ones_like(doc_topics) / len(doc_topics)
            
            # Compute OT coupling
            coupling = ot.emd(
                doc_topics.reshape(-1),
                self.gaussian_ref.reshape(-1),
                self.lda_module.topic_cost_matrix
            )
            
            # Create LOTT embedding
            lot_emb = lot_embedding(coupling, self.lda_module.lda_centers)
            lot_embeddings.append(lot_emb)
        
        return np.array(lot_embeddings)
    
    def rerank_embedding_based(self, query_idx: int, retrieved_doc_ids: List[str], 
                                doc_id_to_idx: Dict, k: int) -> List[str]:
        """
        Rerank using LOTT embedding cosine similarity
        
        Args:
            query_idx: Index of query
            retrieved_doc_ids: List of retrieved doc IDs from BERT
            doc_id_to_idx: Mapping from doc ID to index
            k: Number of top results to return
        
        Returns:
            Reranked doc IDs (top-k)
        """
        query_emb = self.query_lott_embs[query_idx].reshape(1, -1)
        
        # Get embeddings for retrieved docs
        doc_indices = [doc_id_to_idx[doc_id] for doc_id in retrieved_doc_ids]
        doc_embs = self.doc_lott_embs[doc_indices]
        
        # Compute cosine similarities
        similarities = cosine_similarity(query_emb, doc_embs)[0]
        
        # Rerank
        ranked_indices = np.argsort(-similarities)[:k]
        reranked_doc_ids = [retrieved_doc_ids[i] for i in ranked_indices]
        
        return reranked_doc_ids
    
    def rerank_ot_based(self, query_idx: int, retrieved_doc_ids: List[str],
                        doc_id_to_idx: Dict, k: int) -> List[str]:
        """
        Rerank using optimal transport distance between topic distributions
        
        Args:
            query_idx: Index of query
            retrieved_doc_ids: List of retrieved doc IDs from BERT
            doc_id_to_idx: Mapping from doc ID to index
            k: Number of top results to return
        
        Returns:
            Reranked doc IDs (top-k)
        """
        query_topics = self.query_topic_props[query_idx]
        query_topics = query_topics / query_topics.sum() if query_topics.sum() > 0 else query_topics
        
        # Get topic proportions for retrieved docs
        doc_indices = [doc_id_to_idx[doc_id] for doc_id in retrieved_doc_ids]
        
        # Compute OT distances
        ot_distances = []
        for doc_idx in doc_indices:
            doc_topics = self.doc_topic_props[doc_idx]
            doc_topics = doc_topics / doc_topics.sum() if doc_topics.sum() > 0 else doc_topics
            
            # Compute Wasserstein distance
            dist = ot.emd2(
                query_topics,
                doc_topics,
                self.lda_module.topic_cost_matrix
            )
            ot_distances.append(dist)
        
        # Rerank (lower distance = more similar)
        ranked_indices = np.argsort(ot_distances)[:k]
        reranked_doc_ids = [retrieved_doc_ids[i] for i in ranked_indices]
        
        return reranked_doc_ids


if __name__ == "__main__":
    import sys
    from dataset_loaders import load_dataset
    from preprocessing import prepare_bow_data
    from lda_module import LDAModule
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else "trec-covid"
    
    # Load data
    corpus, queries, _ = load_dataset(dataset)
    bow_data, vocab, embeddings, doc_ids = prepare_bow_data(corpus, dataset)
    
    # Load LDA
    lda_module = LDAModule(dataset)
    lda_module.train_or_load(bow_data, embeddings, vocab)
    
    # Prepare reranker
    reranker = LOTTReranker(dataset, lda_module)
    reranker.prepare_documents(bow_data)
    
    # Test
    print("\nReranker ready for use!")
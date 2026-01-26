"""
LDA training and topic inference with caching
"""
import numpy as np
import pickle
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import euclidean_distances
import config


def sparse_ot(weights1, weights2, M):
    """Compute OT distance between sparse distributions"""
    import ot
    
    weights1 = weights1 / weights1.sum()
    weights2 = weights2 / weights2.sum()
    
    active1 = np.where(weights1)[0]
    active2 = np.where(weights2)[0]
    
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    M_reduced = np.ascontiguousarray(M[active1][:, active2])
    
    return ot.emd2(weights_1_active, weights_2_active, M_reduced)


class LDAModule:
    """LDA training with caching"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.cache = config.get_cache_paths(dataset_name)
        self.model = None
        self.topics = None
        self.lda_centers = None
        self.topic_cost_matrix = None
        
    def train_or_load(self, bow_data: np.ndarray, embeddings: np.ndarray, vocab: list):
        """Train LDA or load from cache"""
        
        # Check cache
        if self._load_from_cache():
            return self.topics, self.lda_centers, self.topic_cost_matrix
        
        # Train from scratch
        print(f"Training LDA with {config.N_TOPICS} topics...")
        self.model = LatentDirichletAllocation(
            n_components=config.N_TOPICS,
            max_iter=config.LDA_MAX_ITER,
            random_state=config.LDA_RANDOM_STATE,
            n_jobs=8,
            verbose=1
        )
        
        self.model.fit(bow_data)
        
        # Get topics
        self.topics = self.model.components_
        self.topics = self.topics / self.topics.sum(axis=1, keepdims=True)
        
        # Sparsify topics (keep top-N words)
        print(f"Sparsifying topics to top-{config.N_TOP_WORDS} words...")
        for k in range(config.N_TOPICS):
            to_zero_idx = np.argsort(-self.topics[k])[config.N_TOP_WORDS:]
            self.topics[k][to_zero_idx] = 0
            if self.topics[k].sum() > 0:
                self.topics[k] = self.topics[k] / self.topics[k].sum()
        
        # Compute LDA centers
        self.lda_centers = np.matmul(self.topics, embeddings)
        
        # Compute topic-topic cost matrix
        print("Computing topic-topic cost matrix...")
        cost_embeddings = euclidean_distances(embeddings, embeddings)
        self.topic_cost_matrix = np.zeros((config.N_TOPICS, config.N_TOPICS))
        
        for i in range(config.N_TOPICS):
            if i % 10 == 0:
                print(f"  Progress: {i}/{config.N_TOPICS} topics")
            for j in range(i + 1, config.N_TOPICS):
                self.topic_cost_matrix[i, j] = sparse_ot(
                    self.topics[i], self.topics[j], cost_embeddings
                )
        
        self.topic_cost_matrix = self.topic_cost_matrix + self.topic_cost_matrix.T
        
        # Cache everything
        self._save_to_cache()
        
        print("LDA training complete!")
        return self.topics, self.lda_centers, self.topic_cost_matrix
    
    def infer_topics(self, bow_data: np.ndarray, cache_key: str = 'topic_proportions') -> np.ndarray:
        """Infer topic proportions for documents"""
        
        cache_path = self.cache[cache_key]
        
        # Check cache
        if cache_path.exists():
            print(f"Loading {cache_key} from cache...")
            return np.load(cache_path)
        
        # Infer
        print(f"Inferring topic proportions for {len(bow_data)} documents...")
        topic_proportions = self.model.transform(bow_data)
        
        # Cache
        np.save(cache_path, topic_proportions)
        print(f"Cached to {cache_path}")
        
        return topic_proportions
    
    def _load_from_cache(self) -> bool:
        """Load LDA artifacts from cache"""
        if not self.cache['lda_model'].exists():
            return False
        
        print("Loading LDA from cache...")
        with open(self.cache['lda_model'], 'rb') as f:
            self.model = pickle.load(f)
        self.topics = np.load(self.cache['lda_topics'])
        self.lda_centers = np.load(self.cache['lda_centers'])
        self.topic_cost_matrix = np.load(self.cache['topic_cost_matrix'])
        
        print("LDA loaded from cache!")
        return True
    
    def _save_to_cache(self):
        """Save LDA artifacts to cache"""
        print("Caching LDA artifacts...")
        with open(self.cache['lda_model'], 'rb') as f:
            pickle.dump(self.model, f)
        np.save(self.cache['lda_topics'], self.topics)
        np.save(self.cache['lda_centers'], self.lda_centers)
        np.save(self.cache['topic_cost_matrix'], self.topic_cost_matrix)
        print("LDA artifacts cached!")


if __name__ == "__main__":
    import sys
    from dataset_loaders import load_dataset
    from preprocessing import prepare_bow_data
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else "trec-covid"
    
    # Load data
    corpus, _, _ = load_dataset(dataset)
    bow_data, vocab, embeddings, _ = prepare_bow_data(corpus, dataset)
    
    # Train LDA
    lda_module = LDAModule(dataset)
    topics, centers, costs = lda_module.train_or_load(bow_data, embeddings, vocab)
    
    print(f"Topics shape: {topics.shape}")
    print(f"Centers shape: {centers.shape}")
    print(f"Cost matrix shape: {costs.shape}")
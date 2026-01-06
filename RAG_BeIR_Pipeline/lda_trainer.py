# LDA Model Training and Artifact Persistence
import numpy as np
import pickle
import lda
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import config


def sparse_ot(weights1, weights2, M):
    # Computing OT for Sparse Distributions
    import ot
    
    weights1 = weights1 / weights1.sum()
    weights2 = weights2 / weights2.sum()
    
    active1 = np.where(weights1)[0]
    active2 = np.where(weights2)[0]
    
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    M_reduced = np.ascontiguousarray(M[active1][:, active2])
    
    return ot.emd2(weights_1_active, weights_2_active, M_reduced)


class LDATrainer:
    
    def __init__(self, n_topics=100, n_iterations=1500, random_state=42):
        self.n_topics = n_topics
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.model = None
        self.topics = None
        self.lda_centers = None
        self.topic_cost_matrix = None
        
    def fit(self, bow_data: np.ndarray, embeddings: np.ndarray, vocab: list):
        # Fitting LDA Model to BoW Data
        print(f"Training LDA with {self.n_topics} topics...")
        
        # Initializing and Fitting LDA Model
        self.model = lda.LDA(
            n_topics=self.n_topics,
            n_iter=self.n_iterations,
            random_state=self.random_state
        )
        
        self.model.fit(bow_data)
        
        # Getting Topic-word Distributions
        self.topics = self.model.topic_word_
        
        # Computing LDA Centers (Topics in Embedding Space)
        self.lda_centers = np.matmul(self.topics, embeddings)
        
        print(f"\nLDA Centers shape: {self.lda_centers.shape}")
        
        # Displaying Top words for each Topic
        self._print_top_words(vocab)
        
        return self.topics, self.lda_centers
    
    def _print_top_words(self, vocab: list, n_top_words=10):
        print("\n" + "="*80)
        print("TOP WORDS PER TOPIC")
        print("="*80)
        
        vocab_array = np.array(vocab)
        
        for i, topic_dist in enumerate(self.topics):
            if i >= 10:  # Printing only first 10 Topics
                print(f"... and {self.n_topics - 10} more topics")
                break
            
            topic_words = vocab_array[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
            print(f"Topic {i:2d}: {' '.join(topic_words)}")
        
        print("="*80 + "\n")
    
    def sparsify_topics(self, n_words_keep=20):
        # Reducing Topics to Top-N Words for Sparse Representation
        print(f"Sparsifying topics to top-{n_words_keep} words...")
        
        print(f"Sparsifying topics to top-{n_words_keep} words...")
        
        sparse_topics = self.topics.copy()
        
        for k in range(self.n_topics):
            # Getting Indices to zero out (all except top n_words_keep)
            to_zero_idx = np.argsort(-sparse_topics[k])[n_words_keep:]
            sparse_topics[k][to_zero_idx] = 0
            
            # Renormalizing
            if sparse_topics[k].sum() > 0:
                sparse_topics[k] = sparse_topics[k] / sparse_topics[k].sum()
        
        self.topics = sparse_topics
        print(f"Topics sparsified. Average non-zero words per topic: "
              f"{np.mean((self.topics > 0).sum(axis=1)):.1f}")
    
    def compute_topic_cost_matrix(self, word_embeddings: np.ndarray, p=1):
        # Computing topic-to-topic transport Cost Matrix
        
        print("Computing topic-topic cost matrix...")
        
        # Computing word-to-word Cost
        cost_embeddings = euclidean_distances(word_embeddings, word_embeddings) ** p
        
        # Initializing topic-topic Cost Matrix
        self.topic_cost_matrix = np.zeros((self.n_topics, self.n_topics))
        
        # Computing only Upper Triangle (Symmetric Matrix)
        for i in tqdm(range(self.n_topics), desc="Computing topic costs"):
            for j in range(i + 1, self.n_topics):
                self.topic_cost_matrix[i, j] = sparse_ot(
                    self.topics[i], 
                    self.topics[j], 
                    cost_embeddings
                )

        # Symmetrizing the Cost Matrix
        self.topic_cost_matrix = self.topic_cost_matrix + self.topic_cost_matrix.T
        
        print(f"Topic cost matrix shape: {self.topic_cost_matrix.shape}")
        print(f"Average topic-topic cost: {self.topic_cost_matrix[self.topic_cost_matrix > 0].mean():.4f}")
        
        return self.topic_cost_matrix
    
    def infer_topics(self, bow_data: np.ndarray) -> np.ndarray:
        # Inferring Topic Proportions for New Documents
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Transforming Documents to Topic space
        topic_proportions = self.model.transform(bow_data)
        
        return topic_proportions
    
    def save_artifacts(self):
        # Saving LDA Artifacts
        print("Saving LDA artifacts...")
        
        # Saving LDA model
        with open(config.LDA_MODEL_PATH, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Saving Topics
        np.save(config.LDA_TOPICS_PATH, self.topics)
        
        # Saving LDA Centers
        np.save(config.LDA_CENTERS_PATH, self.lda_centers)
        
        # Saving Topic Cost Matrix
        np.save(config.TOPIC_COST_MATRIX_PATH, self.topic_cost_matrix)
        
        print("Artifacts saved:")
        print(f"  - LDA model: {config.LDA_MODEL_PATH}")
        print(f"  - Topics: {config.LDA_TOPICS_PATH}")
        print(f"  - LDA centers: {config.LDA_CENTERS_PATH}")
        print(f"  - Topic costs: {config.TOPIC_COST_MATRIX_PATH}")
    
    @staticmethod
    def load_artifacts():
        # Loading saved LDA Artifacts
        print("Loading LDA artifacts...")
        
        with open(config.LDA_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        
        topics = np.load(config.LDA_TOPICS_PATH)
        lda_centers = np.load(config.LDA_CENTERS_PATH)
        topic_cost_matrix = np.load(config.TOPIC_COST_MATRIX_PATH)
        
        print("Artifacts loaded successfully!")
        print(f"  - Topics shape: {topics.shape}")
        print(f"  - LDA centers shape: {lda_centers.shape}")
        print(f"  - Topic cost matrix shape: {topic_cost_matrix.shape}")
        
        return model, topics, lda_centers, topic_cost_matrix


def train_and_save_lda(bow_train: np.ndarray, embeddings: np.ndarray, vocab: list):
    trainer = LDATrainer(
        n_topics=config.N_TOPICS,
        n_iterations=config.LDA_ITERATIONS,
        random_state=config.LDA_RANDOM_STATE
    )
    
    # Fitting LDA
    topics, lda_centers = trainer.fit(bow_train, embeddings, vocab)
    
    # Sparsifying Topics
    trainer.sparsify_topics(n_words_keep=config.N_TOP_WORDS)
    
    # Computing Topic Cost Matrix
    trainer.compute_topic_cost_matrix(embeddings, p=config.P_NORM)
    
    # Saving all Artifacts
    trainer.save_artifacts()
    
    return trainer


if __name__ == "__main__":
    from dataset_loader import TRECCovidLoader
    from preprocessing import prepare_bow_data
    
    # Load data
    print("Loading data...")
    train_corpus, test_corpus, _, _ = TRECCovidLoader.load_saved_data()
    
    # Prepare BoW data
    bow_train, bow_test, vocab, embeddings, _, _ = prepare_bow_data(
        train_corpus, test_corpus
    )
    
    # Train and save LDA
    trainer = train_and_save_lda(bow_train, embeddings, vocab)
    
    print("\n" + "="*80)
    print("LDA TRAINING COMPLETE!")
    print("="*80)
import numpy as np
import ot
from typing import List
import config


def makeGaussian1D(size, fwhm=3, center=None):

    center = center if center is not None else size // 2
    
    x = np.arange(size)
    gaussian = np.exp(-4 * np.log(2) * ((x - center)**2) / fwhm**2)
    
    return gaussian / gaussian.sum()


def lot_embedding(coupling_matrix, lda_centers):

    T_mu_sigma = []
    
    for i in range(coupling_matrix.shape[0]):
        source_weight = coupling_matrix[i].sum()
        
        if source_weight == 0:
            y_coor = np.zeros(lda_centers.shape[1], dtype=float)
        else:
            # Get nonzero indices for efficiency
            nonzero_indices = np.where(coupling_matrix[i] > 0)[0]
            
            # Vectorized barycentric projection
            weights = coupling_matrix[i][nonzero_indices]
            positions = lda_centers[nonzero_indices]
            
            y_coor = (weights[:, None] * positions).sum(axis=0) / source_weight
        
        T_mu_sigma.append(y_coor)
    
    return np.array(T_mu_sigma).flatten()


def create_lot_embeddings(topic_proportions: np.ndarray, 
                          reference_dist: np.ndarray,
                          lda_centers: np.ndarray,
                          topic_cost_matrix: np.ndarray,
                          show_progress: bool = True) -> List[np.ndarray]:

    lot_embeddings = []
    
    if show_progress:
        print(f"Creating LOTT embeddings for {len(topic_proportions)} documents...")
    
    for i in range(len(topic_proportions)):
        if show_progress and i % 5000 == 0:
            print(f"  Progress: {i}/{len(topic_proportions)} documents")
        
        # Get topic distribution for this document
        doc_topics = topic_proportions[i].reshape(-1)
        
        # Normalize if needed
        if doc_topics.sum() > 0:
            doc_topics = doc_topics / doc_topics.sum()
        else:
            # If no topics, use uniform distribution
            doc_topics = np.ones_like(doc_topics) / len(doc_topics)
        
        # Compute optimal transport plan
        coupling = ot.emd(
            doc_topics.reshape(-1), 
            reference_dist.reshape(-1), 
            topic_cost_matrix
        )
        
        # Create LOT embedding
        lot_emb = lot_embedding(coupling, lda_centers)
        lot_embeddings.append(lot_emb)
    
    return np.array(lot_embeddings)


class LOTTEmbeddingGenerator:
    def __init__(self, lda_model, lda_centers, topic_cost_matrix, gaussian_fwhm=25):
        self.lda_model = lda_model
        self.lda_centers = lda_centers
        self.topic_cost_matrix = topic_cost_matrix
        self.gaussian_fwhm = gaussian_fwhm
        self.n_topics = lda_centers.shape[0]
        
        # Create reference Gaussian distribution
        self.reference_dist = makeGaussian1D(
            size=self.n_topics, 
            fwhm=self.gaussian_fwhm
        )
    
    def infer_topics(self, bow_data: np.ndarray) -> np.ndarray:
        print("Inferring topic proportions...")
        topic_proportions = self.lda_model.transform(bow_data)
        return topic_proportions
    
    def generate_embeddings(self, topic_proportions: np.ndarray, 
                           show_progress: bool = True) -> np.ndarray:
        embeddings = create_lot_embeddings(
            topic_proportions,
            self.reference_dist,
            self.lda_centers,
            self.topic_cost_matrix,
            show_progress=show_progress
        )
        
        return embeddings
    
    def generate_from_bow(self, bow_data: np.ndarray, 
                         show_progress: bool = True) -> tuple:
        # Infer topics
        topic_proportions = self.infer_topics(bow_data)
        
        # Generate LOTT embeddings
        embeddings = self.generate_embeddings(topic_proportions, show_progress)
        
        return embeddings, topic_proportions


def generate_and_save_lott_embeddings(bow_all: np.ndarray):
    # Loading LDA Artifacts
    from lda_trainer import LDATrainer
    lda_model, topics, lda_centers, topic_cost_matrix = LDATrainer.load_artifacts()
    
    # Initializing Generator
    generator = LOTTEmbeddingGenerator(
        lda_model=lda_model,
        lda_centers=lda_centers,
        topic_cost_matrix=topic_cost_matrix,
        gaussian_fwhm=config.GAUSSIAN_FWHM
    )
    
    # Generating Embeddings for all Documents
    print("\n" + "="*80)
    print("GENERATING LOTT EMBEDDINGS FOR ALL DOCUMENTS")
    print("="*80)
    all_embeddings, all_topics = generator.generate_from_bow(bow_all, show_progress=True)
    
    # Saving Embeddings
    np.save(config.LOTT_EMBEDDINGS, all_embeddings)
    np.save(config.TOPIC_PROPORTIONS, all_topics)
    print(f"Saved embeddings to {config.LOTT_EMBEDDINGS}")
    print(f"Shape: {all_embeddings.shape}")
    
    return all_embeddings, all_topics

def load_lott_embeddings():
    print("Loading LOTT embeddings...")
    
    embeddings = np.load(config.LOTT_EMBEDDINGS)
    topics = np.load(config.TOPIC_PROPORTIONS)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    print(f"Loaded topics: {topics.shape}")
    
    return embeddings, topics


if __name__ == "__main__":
    from preprocessing import TextPreprocessor
    import pickle
    
    # Load saved BoW data (you'll need to save these in preprocessing step)
    print("This script requires preprocessed BoW data.")
    print("Run the full pipeline script to generate LOTT embeddings.")
    
    # Example usage:
    # bow_train = np.load('bow_train.npy')
    # bow_test = np.load('bow_test.npy')
    # train_emb, test_emb, train_topics, test_topics = generate_and_save_lott_embeddings(bow_train, bow_test)
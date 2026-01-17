# LOTT Embedding Generation
import numpy as np
import ot
# from tqdm import tqdm
from typing import List
import config
from sklearn.decomposition import PCA

def makeGaussian1D(size, fwhm=3, center=None):
    # Generating a 1D Gaussian distribution.
    center = center if center is not None else size // 2
    
    x = np.arange(size)
    gaussian = np.exp(-4 * np.log(2) * ((x - center)**2) / fwhm**2)
    
    return gaussian / gaussian.sum()


def lot_embedding(coupling_matrix, lda_centers):
    # Computing LOT Embedding from Optimal Transport Coupling
    T_mu_sigma = []
    
    for i in range(coupling_matrix.shape[0]):
        source_weight = coupling_matrix[i].sum()
        
        if source_weight == 0:
            y_coor = np.zeros(lda_centers.shape[1], dtype=float)
        else:
            # Getting nonzero Indices for Efficiency
            nonzero_indices = np.where(coupling_matrix[i] > 0)[0]
            
            # Vectorized Barycentric Projection
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
    # LOTT Embeddings for multiple Documents
    lot_embeddings = []
    
    # iterator = tqdm(range(len(topic_proportions)), desc="Creating LOTT embeddings") \
    #            if show_progress else range(len(topic_proportions))
    
    # for i in iterator:
    if show_progress:
        print(f"Creating LOTT embeddings for {len(topic_proportions)} documents...")
    
    for i in range(len(topic_proportions)):
        if show_progress and i % 5000 == 0:
            print(f"  Progress: {i}/{len(topic_proportions)} documents")
        # Topic distribution for current Document
        doc_topics = topic_proportions[i].reshape(-1)
        
        # Normalized if needed
        if doc_topics.sum() > 0:
            doc_topics = doc_topics / doc_topics.sum()
        else:
            # If no topics, using Uniform distribution
            doc_topics = np.ones_like(doc_topics) / len(doc_topics)
        
        # Computing Optimal Transport plan
        coupling = ot.emd(
            doc_topics.reshape(-1), 
            reference_dist.reshape(-1), 
            topic_cost_matrix
        )
        
        # Creating LOT Embedding
        lot_emb = lot_embedding(coupling, lda_centers)
        lot_embeddings.append(lot_emb)
    
    return np.array(lot_embeddings)


class LOTTEmbeddingGenerator:
    # Generating LOTT Embeddings for Documents
    
    def __init__(self, lda_model, lda_centers, topic_cost_matrix, gaussian_fwhm=25):
        self.lda_model = lda_model
        self.lda_centers = lda_centers
        self.topic_cost_matrix = topic_cost_matrix
        self.gaussian_fwhm = gaussian_fwhm
        self.n_topics = lda_centers.shape[0]
        
        # Creating reference Gaussian distribution
        self.reference_dist = makeGaussian1D(
            size=self.n_topics, 
            fwhm=self.gaussian_fwhm
        )
    
    def infer_topics(self, bow_data: np.ndarray) -> np.ndarray:
        # Inferring Topic Proportions for BoW data
        print("Inferring topic proportions...")
        topic_proportions = self.lda_model.transform(bow_data)
        return topic_proportions
    
    def generate_embeddings(self, topic_proportions: np.ndarray, 
                           show_progress: bool = True) -> np.ndarray:
        # Generating LOTT Embeddings from topic proportions
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
        # Generating LOTT Embeddings directly from BoW data
        # Inferring Topics
        topic_proportions = self.infer_topics(bow_data)
        
        embeddings = self.generate_embeddings(topic_proportions, show_progress)
        
        return embeddings, topic_proportions


# def generate_and_save_lott_embeddings(bow_train: np.ndarray, bow_test: np.ndarray):
#     # Generating and saving LOTT Embeddings for train and test sets
#     # Loading LDA Artifacts
#     from lda_trainer import LDATrainer
#     lda_model, topics, lda_centers, topic_cost_matrix = LDATrainer.load_artifacts()
    
#     # Initializing Generator
#     generator = LOTTEmbeddingGenerator(
#         lda_model=lda_model,
#         lda_centers=lda_centers,
#         topic_cost_matrix=topic_cost_matrix,
#         gaussian_fwhm=config.GAUSSIAN_FWHM
#     )
    
#     # Generating Train embeddings
#     print("\n" + "="*80)
#     print("GENERATING TRAINING LOTT EMBEDDINGS")
#     print("="*80)
#     train_embeddings, train_topics = generator.generate_from_bow(bow_train, show_progress=True)
    
#     # Saving Train Embeddings and Topics
#     np.save(config.LOTT_TRAIN_EMBEDDINGS, train_embeddings)
#     np.save(config.TOPIC_PROPORTIONS_TRAIN, train_topics)
#     print(f"Saved train embeddings to {config.LOTT_TRAIN_EMBEDDINGS}")
#     print(f"Shape: {train_embeddings.shape}")
    
#     # Generating Test Embeddings
#     print("\n" + "="*80)
#     print("GENERATING TEST LOTT EMBEDDINGS")
#     print("="*80)
#     test_embeddings, test_topics = generator.generate_from_bow(bow_test, show_progress=True)
    
#     # Saving Test Embeddings and Topics
#     np.save(config.LOTT_TEST_EMBEDDINGS, test_embeddings)
#     np.save(config.TOPIC_PROPORTIONS_TEST, test_topics)
#     print(f"Saved test embeddings to {config.LOTT_TEST_EMBEDDINGS}")
#     print(f"Shape: {test_embeddings.shape}")
    
#     return train_embeddings, test_embeddings, train_topics, test_topics

    # # PCA Compression
    # print("\nCompressing LOTT embeddings with PCA...")
    # print(f"Original dimension: {train_embeddings.shape[1]}")
    
    # pca = PCA(n_components=384)  # Matching BERT dimension
    # train_embeddings_compressed = pca.fit_transform(train_embeddings)
    # test_embeddings_compressed = pca.transform(test_embeddings)
    
    # print(f"Compressed dimension: {train_embeddings_compressed.shape[1]}")
    # print(f"Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    
    # # Saving Compressed Embeddings
    # np.save(config.LOTT_TRAIN_EMBEDDINGS, train_embeddings_compressed)
    # np.save(config.LOTT_TEST_EMBEDDINGS, test_embeddings_compressed)
    
    # # Also save the PCA model for query compression
    # import pickle
    # pca_path = config.MODELS_DIR / "lott_pca.pkl"
    # with open(pca_path, 'wb') as f:
    #     pickle.dump(pca, f)
    # print(f"PCA model saved to {pca_path}")
    
    # # Save topic proportions
    # np.save(config.TOPIC_PROPORTIONS_TRAIN, train_topics)
    # np.save(config.TOPIC_PROPORTIONS_TEST, test_topics)
    
    # print(f"Saved train embeddings to {config.LOTT_TRAIN_EMBEDDINGS}")
    # print(f"Shape: {train_embeddings_compressed.shape}")
    # print(f"Saved test embeddings to {config.LOTT_TEST_EMBEDDINGS}")
    # print(f"Shape: {test_embeddings_compressed.shape}")
    
    # return train_embeddings_compressed, test_embeddings_compressed, train_topics, test_topics

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
    np.save(config.LOTT_TRAIN_EMBEDDINGS, all_embeddings)
    np.save(config.TOPIC_PROPORTIONS_TRAIN, all_topics)
    print(f"Saved embeddings to {config.LOTT_TRAIN_EMBEDDINGS}")
    print(f"Shape: {all_embeddings.shape}")
    
    return all_embeddings, all_topics

def load_lott_embeddings():
    # Loading saved LOTT embeddings
    print("Loading LOTT embeddings...")
    
    # train_embeddings = np.load(config.LOTT_TRAIN_EMBEDDINGS)
    # test_embeddings = np.load(config.LOTT_TEST_EMBEDDINGS)
    # train_topics = np.load(config.TOPIC_PROPORTIONS_TRAIN)
    # test_topics = np.load(config.TOPIC_PROPORTIONS_TEST)

    all_embeddings = np.load(config.LOTT_TRAIN_EMBEDDINGS)
    all_topics = np.load(config.TOPIC_PROPORTIONS_TRAIN)
    
    # print(f"Loaded train embeddings: {train_embeddings.shape}")
    # print(f"Loaded test embeddings: {test_embeddings.shape}")
    # print(f"Loaded train topics: {train_topics.shape}")
    # print(f"Loaded test topics: {test_topics.shape}")

    print(f"Loaded all embeddings: {all_embeddings.shape}")
    print(f"Loaded all topics: {all_topics.shape}")
    
    # return train_embeddings, test_embeddings, train_topics, test_topics
    return all_embeddings, all_topics


if __name__ == "__main__":
    from preprocessing import TextPreprocessor
    import pickle
    
    # Loading saved BoW data
    print("This script requires preprocessed BoW data.")
    print("Run the full pipeline script to generate LOTT embeddings.")
    
    # Example usage:
    # bow_train = np.load('bow_train.npy')
    # bow_test = np.load('bow_test.npy')
    # train_emb, test_emb, train_topics, test_topics = generate_and_save_lott_embeddings(bow_train, bow_test)
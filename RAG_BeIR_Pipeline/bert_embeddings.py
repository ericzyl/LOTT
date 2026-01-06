import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from tqdm import tqdm
import config

# Sentence Transformer for generating BERT Embeddings for Documents and Queries
class BERTEmbeddingGenerator:
    
    def __init__(self, model_name=None, batch_size=32):
        self.model_name = model_name or config.BERT_MODEL
        self.batch_size = batch_size
        self.model = None
        
    def load_model(self):
        print(f"Loading BERT model: {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        print("Model loaded successfully!")
        print(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        
    def generate_embeddings(self, texts: List[str], show_progress=True) -> np.ndarray:
        # Generating Embeddings for a List of Texts
        if self.model is None:
            self.load_model()
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def generate_corpus_embeddings(self, corpus: Dict[str, Dict], 
                                   doc_ids: List[str] = None) -> np.ndarray:
        # Embeddings for entire Corpus
        if doc_ids is None:
            doc_ids = list(corpus.keys())
        
        # Extracting Texts in order
        texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
        
        print(f"Generating BERT embeddings for {len(texts)} documents...")
        embeddings = self.generate_embeddings(texts, show_progress=True)
        
        return embeddings
    
    def generate_query_embeddings(self, queries: Dict[str, str]) -> tuple:
        # Embeddings for Queries
        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        print(f"Generating BERT embeddings for {len(query_texts)} queries...")
        embeddings = self.generate_embeddings(query_texts, show_progress=True)
        
        return embeddings, query_ids


def generate_and_save_bert_embeddings(train_corpus: Dict, test_corpus: Dict,
                                      train_doc_ids: List[str], test_doc_ids: List[str]):
    generator = BERTEmbeddingGenerator(batch_size=config.BERT_BATCH_SIZE)
    generator.load_model()
    
    print("\n" + "="*80)
    print("GENERATING TRAINING EMBEDDINGS")
    print("="*80)
    train_embeddings = generator.generate_corpus_embeddings(train_corpus, train_doc_ids)
    
    np.save(config.BERT_TRAIN_EMBEDDINGS, train_embeddings)
    print(f"Saved train embeddings to {config.BERT_TRAIN_EMBEDDINGS}")
    print(f"Shape: {train_embeddings.shape}")
    
    print("\n" + "="*80)
    print("GENERATING TEST EMBEDDINGS")
    print("="*80)
    test_embeddings = generator.generate_corpus_embeddings(test_corpus, test_doc_ids)
    
    np.save(config.BERT_TEST_EMBEDDINGS, test_embeddings)
    print(f"Saved test embeddings to {config.BERT_TEST_EMBEDDINGS}")
    print(f"Shape: {test_embeddings.shape}")
    
    return train_embeddings, test_embeddings


def load_bert_embeddings():
    print("Loading BERT embeddings...")
    
    train_embeddings = np.load(config.BERT_TRAIN_EMBEDDINGS)
    test_embeddings = np.load(config.BERT_TEST_EMBEDDINGS)
    
    print(f"Loaded train embeddings: {train_embeddings.shape}")
    print(f"Loaded test embeddings: {test_embeddings.shape}")
    
    return train_embeddings, test_embeddings


if __name__ == "__main__":
    import pickle
    from dataset_loader import TRECCovidLoader
    
    print("Loading corpus...")
    train_corpus, test_corpus, queries, _ = TRECCovidLoader.load_saved_data()
    
    # Loading Vocabulary to get Document ordering
    with open(config.VOCAB_PATH, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # Using all document IDs in Corpus order for now
    train_doc_ids = list(train_corpus.keys())
    test_doc_ids = list(test_corpus.keys())
    
    train_emb, test_emb = generate_and_save_bert_embeddings(
        train_corpus, test_corpus, train_doc_ids, test_doc_ids
    )
    
    print("\n" + "="*80)
    print("BERT EMBEDDING GENERATION COMPLETE!")
    print("="*80)
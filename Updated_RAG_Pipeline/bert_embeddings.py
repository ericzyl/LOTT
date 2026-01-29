import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List
from tqdm import tqdm
import config


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

        if self.model is None:
            self.load_model()
        
        # Generating Embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def generate_corpus_embeddings(self, corpus: Dict[str, Dict], 
                                   doc_ids: List[str] = None) -> np.ndarray:

        if doc_ids is None:
            doc_ids = list(corpus.keys())
        
        # Extracting texts in order
        texts = [corpus[doc_id]['text'] for doc_id in doc_ids]
        
        print(f"Generating BERT embeddings for {len(texts)} documents...")
        embeddings = self.generate_embeddings(texts, show_progress=True)
        
        return embeddings
    
    def generate_query_embeddings(self, queries: Dict[str, str]) -> tuple:

        query_ids = list(queries.keys())
        query_texts = [queries[qid] for qid in query_ids]
        
        print(f"Generating BERT embeddings for {len(query_texts)} queries...")
        embeddings = self.generate_embeddings(query_texts, show_progress=True)
        
        return embeddings, query_ids


def generate_and_save_bert_embeddings(corpus: Dict, doc_ids: List[str]):

    generator = BERTEmbeddingGenerator(batch_size=config.BERT_BATCH_SIZE)
    generator.load_model()
    
    # Generating embeddings
    print("\n" + "="*80)
    print("GENERATING BERT EMBEDDINGS FOR ALL DOCUMENTS")
    print("="*80)
    embeddings = generator.generate_corpus_embeddings(corpus, doc_ids)
    
    # Saving embeddings
    np.save(config.BERT_EMBEDDINGS, embeddings)
    print(f"Saved embeddings to {config.BERT_EMBEDDINGS}")
    print(f"Shape: {embeddings.shape}")
    
    return embeddings


def load_bert_embeddings():
    """Load saved BERT embeddings"""
    print("Loading BERT embeddings...")
    
    embeddings = np.load(config.BERT_EMBEDDINGS)
    
    print(f"Loaded embeddings: {embeddings.shape}")
    
    return embeddings


if __name__ == "__main__":
    import pickle
    from dataset_loader import TRECCovidLoader
    
    # Loading corpus
    print("Loading corpus...")
    corpus, queries, _ = TRECCovidLoader.load_saved_data()
    
    # Using all document IDs in corpus order
    doc_ids = list(corpus.keys())
    
    embeddings = generate_and_save_bert_embeddings(corpus, doc_ids)
    
    print("\n" + "="*80)
    print("BERT EMBEDDING GENERATION COMPLETE!")
    print("="*80)
"""
Preprocessing utilities for text data
Handles vocabulary building, BoW conversion, and GloVe loading
"""
import numpy as np
import pickle
from collections import Counter
from typing import Dict, List, Tuple
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import config

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextPreprocessor:
    """Handles text preprocessing for BoW representation"""
    
    def __init__(self, min_word_length=3, max_vocab_size=10000):
        self.min_word_length = min_word_length
        self.max_vocab_size = max_vocab_size
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vocab = None
        self.word_to_idx = None
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize and clean text"""
        # Lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Filter by length and stop words
        words = [w for w in words if len(w) >= self.min_word_length and w not in self.stop_words]
        
        # Lemmatize
        words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return words
    
    def build_vocabulary(self, corpus: Dict[str, Dict], glove_path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
        """Build vocabulary from corpus and load GloVe embeddings"""
        print("Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        print("Counting words in corpus...")
        for idx, (doc_id, doc_data) in enumerate(corpus.items()):
            if idx % 5000 == 0:
                print(f"  Processed {idx}/{len(corpus)} documents")
            words = self.tokenize(doc_data['text'])
            word_counts.update(words)
        
        print(f"Total unique words before filtering: {len(word_counts)}")
        
        # Load GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}...")
        glove_embeddings = {}
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            total_lines = len(lines)
            for idx, line in enumerate(lines):
                if idx % 50000 == 0:
                    print(f"  Loaded {idx}/{total_lines} embeddings")
                try:
                    parts = line.strip().split()
                    
                    # Skipping Empty lines
                    if len(parts) < 2:
                        continue
                    
                    word = parts[0]
                    
                    # Trying to convert remaining parts to floats
                    # Skipping the line if any part can't be converted
                    try:
                        embedding = np.array([float(x) for x in parts[1:]])
                    except ValueError:
                        # Skipping malformed lines
                        continue
                    
                    # Sanity check: embedding should have expected dimension
                    if len(embedding) != config.GLOVE_DIM:
                        continue
                    
                    glove_embeddings[word] = embedding
                    
                except Exception as e:
                    # Skipping any problematic lines
                    if idx < 10:  # Printing first few errors for debugging
                        print(f"  Warning: Skipping line {idx}: {e}")
                    continue
        
        print(f"Loaded {len(glove_embeddings)} GloVe embeddings")
        
        # Keep only words that exist in GloVe and are most frequent
        valid_words = [(word, count) for word, count in word_counts.items() 
                       if word in glove_embeddings]
        
        # Sort by frequency and take top words
        valid_words.sort(key=lambda x: x[1], reverse=True)
        valid_words = valid_words[:self.max_vocab_size]
        
        # Build vocabulary
        self.vocab = [word for word, _ in valid_words]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Get embeddings for vocabulary
        vocab_embeddings = {word: glove_embeddings[word] for word in self.vocab}
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        
        return self.vocab, vocab_embeddings
    
    def text_to_bow(self, text: str) -> np.ndarray:
        """Convert text to bag-of-words vector"""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        words = self.tokenize(text)
        bow = np.zeros(len(self.vocab), dtype=np.int32)
        
        for word in words:
            if word in self.word_to_idx:
                bow[self.word_to_idx[word]] += 1
        
        return bow
    
    def corpus_to_bow(self, corpus: Dict[str, Dict]) -> Tuple[np.ndarray, List[str]]:
        """Convert entire corpus to BoW matrix"""
        print("Converting corpus to BoW representation...")
        
        doc_ids = list(corpus.keys())
        bow_matrix = np.zeros((len(doc_ids), len(self.vocab)), dtype=np.int32)
        
        for i, doc_id in enumerate(doc_ids):
            if i % 5000 == 0:
                print(f"  Converted {i}/{len(doc_ids)} documents")
            text = corpus[doc_id]['text']
            bow_matrix[i] = self.text_to_bow(text)
        
        # Remove documents with no words in vocabulary
        valid_docs = bow_matrix.sum(axis=1) > 0
        bow_matrix = bow_matrix[valid_docs]
        doc_ids = [doc_id for i, doc_id in enumerate(doc_ids) if valid_docs[i]]
        
        print(f"Valid documents after BoW conversion: {len(doc_ids)}")
        
        return bow_matrix, doc_ids
    
    def save_vocabulary(self):
        """Save vocabulary and word mappings"""
        vocab_data = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx
        }
        with open(config.VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {config.VOCAB_PATH}")
    
    def load_vocabulary(self):
        """Load saved vocabulary"""
        with open(config.VOCAB_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        self.vocab = vocab_data['vocab']
        self.word_to_idx = vocab_data['word_to_idx']
        print(f"Vocabulary loaded: {len(self.vocab)} words")


def prepare_bow_data(corpus: Dict):
    """
    Main function to prepare BoW data
    Returns: bow_data, vocab, embeddings, doc_ids
    """
    preprocessor = TextPreprocessor(
        min_word_length=3,
        max_vocab_size=10000
    )
    
    # Build vocabulary from corpus
    vocab, vocab_embeddings = preprocessor.build_vocabulary(corpus, str(config.GLOVE_PATH))
    
    # Save vocabulary
    preprocessor.save_vocabulary()
    
    # Convert to BoW
    bow_data, doc_ids = preprocessor.corpus_to_bow(corpus)
    
    # Convert embeddings dict to array
    embeddings = np.array([vocab_embeddings[word] for word in vocab])
    
    # Save embeddings
    np.save(config.WORD_EMBEDDINGS_PATH, embeddings)
    print(f"Word embeddings saved to {config.WORD_EMBEDDINGS_PATH}")
    
    return bow_data, vocab, embeddings, doc_ids


if __name__ == "__main__":
    # Load saved corpus
    from dataset_loader import TRECCovidLoader
    
    corpus, _, _ = TRECCovidLoader.load_saved_data()
    
    # Prepare BoW data
    bow_data, vocab, embeddings, doc_ids = prepare_bow_data(corpus)
    
    print("\nPreprocessing complete!")
    print(f"BoW shape: {bow_data.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
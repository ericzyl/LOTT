# Preprocessing Utilities for Text
import numpy as np
import pickle
from collections import Counter
from typing import Dict, List, Tuple
# from tqdm import tqdm
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import config

#NLTK Requirements
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
    # Handling Text Preprocessing for BoW Representation
    
    def __init__(self, min_word_length=3, max_vocab_size=10000):
        self.min_word_length = min_word_length
        self.max_vocab_size = max_vocab_size
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.vocab = None
        self.word_to_idx = None
        
    def tokenize(self, text: str) -> List[str]:
        # Tokenizing and Cleaning Text
        # Lowercasing
        text = text.lower()
        
        # Removing special characters and digits
        text = re.sub(r'[^a-z\s]', ' ', text)
        
        # Splitting into words
        words = text.split()
        
        # Filtering by length and stop words
        words = [w for w in words if len(w) >= self.min_word_length and w not in self.stop_words]
        
        # Lemmatizing
        words = [self.lemmatizer.lemmatize(w) for w in words]
        
        return words
    
    def build_vocabulary(self, corpus: Dict[str, Dict], glove_path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
        # Building vocabulary from corpus and load GloVe embeddings
        print("Building vocabulary...")
        
        # Counting word frequencies
        word_counts = Counter()
        # for doc_id, doc_data in tqdm(corpus.items(), desc="Counting words"):
        print("Counting words in corpus...")
        for idx, (doc_id, doc_data) in enumerate(corpus.items()):
            if idx % 5000 == 0:
                print(f"  Processed {idx}/{len(corpus)} documents")
            words = self.tokenize(doc_data['text'])
            word_counts.update(words)
        
        print(f"Total unique words before filtering: {len(word_counts)}")
        
        # Loading GloVe embeddings
        print(f"Loading GloVe embeddings from {glove_path}...")
        glove_embeddings = {}
        
        with open(glove_path, 'r', encoding='utf-8') as f:
            # for line in tqdm(f, desc="Loading GloVe"):
            lines = f.readlines()
            total_lines = len(lines)
            for idx, line in enumerate(lines):
                if idx % 50000 == 0:
                    print(f"  Loaded {idx}/{total_lines} embeddings")
                parts = line.strip().split()
                word = parts[0]
                embedding = np.array([float(x) for x in parts[1:]])
                glove_embeddings[word] = embedding
        
        print(f"Loaded {len(glove_embeddings)} GloVe embeddings")
        
        # Keeping only words that exist in GloVe and are most frequent
        valid_words = [(word, count) for word, count in word_counts.items() 
                       if word in glove_embeddings]
        
        # Sorting by frequency and taking top words
        valid_words.sort(key=lambda x: x[1], reverse=True)
        valid_words = valid_words[:self.max_vocab_size]
        
        # Building Vocabulary
        self.vocab = [word for word, _ in valid_words]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        
        # Getting embeddings for vocabulary
        vocab_embeddings = {word: glove_embeddings[word] for word in self.vocab}
        
        print(f"Final vocabulary size: {len(self.vocab)}")
        
        return self.vocab, vocab_embeddings
    
    def text_to_bow(self, text: str) -> np.ndarray:
        # Converting text to BoW vector
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Call build_vocabulary first.")
        
        words = self.tokenize(text)
        bow = np.zeros(len(self.vocab), dtype=np.int32)
        
        for word in words:
            if word in self.word_to_idx:
                bow[self.word_to_idx[word]] += 1
        
        return bow
    
    def corpus_to_bow(self, corpus: Dict[str, Dict]) -> Tuple[np.ndarray, List[str]]:
        # Converting entire corpus to BoW matrix
        print("Converting corpus to BoW representation...")
        
        doc_ids = list(corpus.keys())
        bow_matrix = np.zeros((len(doc_ids), len(self.vocab)), dtype=np.int32)
        
        # for i, doc_id in enumerate(tqdm(doc_ids, desc="Converting to BoW")):
        for i, doc_id in enumerate(doc_ids):
            if i % 5000 == 0:
                print(f"  Converted {i}/{len(doc_ids)} documents")
            text = corpus[doc_id]['text']
            bow_matrix[i] = self.text_to_bow(text)
        
        # Removing documents with no words in vocabulary
        valid_docs = bow_matrix.sum(axis=1) > 0
        bow_matrix = bow_matrix[valid_docs]
        doc_ids = [doc_id for i, doc_id in enumerate(doc_ids) if valid_docs[i]]
        
        print(f"Valid documents after BoW conversion: {len(doc_ids)}")
        
        return bow_matrix, doc_ids
    
    def save_vocabulary(self):
        # Saving vocabulary and word mappings
        vocab_data = {
            'vocab': self.vocab,
            'word_to_idx': self.word_to_idx
        }
        with open(config.VOCAB_PATH, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {config.VOCAB_PATH}")
    
    def load_vocabulary(self):
        # Loading saved vocabulary
        with open(config.VOCAB_PATH, 'rb') as f:
            vocab_data = pickle.load(f)
        self.vocab = vocab_data['vocab']
        self.word_to_idx = vocab_data['word_to_idx']
        print(f"Vocabulary loaded: {len(self.vocab)} words")


def prepare_bow_data(train_corpus: Dict, test_corpus: Dict):
    preprocessor = TextPreprocessor(
        min_word_length=3,
        max_vocab_size=10000
    )
    
    vocab, vocab_embeddings = preprocessor.build_vocabulary(train_corpus, str(config.GLOVE_PATH))
    
    preprocessor.save_vocabulary()
    
    bow_train, train_doc_ids = preprocessor.corpus_to_bow(train_corpus)
    bow_test, test_doc_ids = preprocessor.corpus_to_bow(test_corpus)
    
    embeddings = np.array([vocab_embeddings[word] for word in vocab])
    
    np.save(config.WORD_EMBEDDINGS_PATH, embeddings)
    print(f"Word embeddings saved to {config.WORD_EMBEDDINGS_PATH}")
    
    return bow_train, bow_test, vocab, embeddings, train_doc_ids, test_doc_ids


if __name__ == "__main__":
    from dataset_loader import TRECCovidLoader
    
    train_corpus, test_corpus, _, _ = TRECCovidLoader.load_saved_data()
    
    bow_train, bow_test, vocab, embeddings, train_ids, test_ids = prepare_bow_data(
        train_corpus, test_corpus
    )
    
    print("\nPreprocessing complete!")
    print(f"BoW train shape: {bow_train.shape}")
    print(f"BoW test shape: {bow_test.shape}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
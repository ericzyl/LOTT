import pickle
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import config

for _resource, _path in [
    ('corpora/stopwords', 'stopwords'),
    ('corpora/wordnet',   'wordnet'),
    ('corpora/omw-1.4',   'omw-1.4'),
]:
    try:
        nltk.data.find(_resource)
    except LookupError:
        nltk.download(_path, quiet=True)


class TextPreprocessor:

    def __init__(self, min_word_length: int = 3, max_vocab_size: int = 10_000):
        self.min_word_length = min_word_length
        self.max_vocab_size  = max_vocab_size
        self.stop_words  = set(stopwords.words('english'))
        self.lemmatizer  = WordNetLemmatizer()
        self.vocab:       List[str]       = None
        self.word_to_idx: Dict[str, int]  = None

    def tokenize(self, text: str) -> List[str]:
        text  = text.lower()
        text  = re.sub(r'[^a-z\s]', ' ', text)
        words = text.split()
        words = [w for w in words
                 if len(w) >= self.min_word_length and w not in self.stop_words]
        words = [self.lemmatizer.lemmatize(w) for w in words]
        return words

    def build_vocabulary(
        self, corpus: Dict[str, Dict], glove_path: str
    ) -> Tuple[List[str], Dict[str, np.ndarray]]:

        print("Building vocabulary from corpus...")
        word_counts: Counter = Counter()
        for idx, (_, doc_data) in enumerate(corpus.items()):
            if idx % 10_000 == 0:
                print(f"  Counting words: {idx}/{len(corpus)}")
            word_counts.update(self.tokenize(doc_data['text']))
        print(f"Unique words before GloVe filter: {len(word_counts)}")

        print(f"Loading GloVe from {glove_path} ...")
        glove: Dict[str, np.ndarray] = {}
        with open(glove_path, 'r', encoding='utf-8') as fh:
            for idx, line in enumerate(fh):
                if idx % 100_000 == 0:
                    print(f"  GloVe lines loaded: {idx}")
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                word = parts[0]
                try:
                    vec = np.array([float(x) for x in parts[1:]])
                except ValueError:
                    continue
                if len(vec) != config.GLOVE_DIM:
                    continue
                glove[word] = vec
        print(f"GloVe vectors loaded: {len(glove)}")

        valid = [(w, c) for w, c in word_counts.items() if w in glove]
        valid.sort(key=lambda x: x[1], reverse=True)
        valid = valid[: self.max_vocab_size]

        self.vocab       = [w for w, _ in valid]
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        vocab_embs       = {w: glove[w] for w in self.vocab}

        print(f"Final vocabulary size: {len(self.vocab)}")
        return self.vocab, vocab_embs

    def text_to_bow(self, text: str) -> np.ndarray:
        if self.vocab is None:
            raise RuntimeError("Call build_vocabulary first.")
        bow = np.zeros(len(self.vocab), dtype=np.int32)
        for w in self.tokenize(text):
            if w in self.word_to_idx:
                bow[self.word_to_idx[w]] += 1
        return bow

    def corpus_to_bow(self, corpus: Dict[str, Dict]) -> Tuple[np.ndarray, List[str]]:
        print(f"Converting {len(corpus)} documents to BoW...")
        doc_ids = list(corpus.keys())
        matrix  = np.zeros((len(doc_ids), len(self.vocab)), dtype=np.int32)
        for i, doc_id in enumerate(doc_ids):
            if i % 10_000 == 0:
                print(f"  BoW conversion: {i}/{len(doc_ids)}")
            matrix[i] = self.text_to_bow(corpus[doc_id]['text'])
        valid    = matrix.sum(axis=1) > 0
        matrix   = matrix[valid]
        doc_ids  = [d for d, keep in zip(doc_ids, valid) if keep]
        print(f"Valid documents after BoW: {len(doc_ids)}")
        return matrix, doc_ids


def prepare_bow_data(corpus: Dict, dataset_name: str) -> Tuple:
    cache = config.get_cache_paths(dataset_name)

    if cache['bow_data'].exists() and cache['vocab'].exists():
        print("Loading BoW data from cache...")
        bow_data = np.load(cache['bow_data'])
        with open(cache['vocab'], 'rb') as f:
            vocab_data = pickle.load(f)
        vocab      = vocab_data['vocab']
        embeddings = np.load(cache['word_embeddings'])
        with open(cache['doc_ids'], 'rb') as f:
            doc_ids = pickle.load(f)
        print(f"Cached BoW: {bow_data.shape}, vocab: {len(vocab)}, embs: {embeddings.shape}")
        return bow_data, vocab, embeddings, doc_ids

    preprocessor = TextPreprocessor(
        min_word_length=config.MIN_WORD_LENGTH,
        max_vocab_size=config.MAX_VOCAB_SIZE,
    )
    vocab, vocab_embs = preprocessor.build_vocabulary(corpus, str(config.GLOVE_PATH))
    bow_data, doc_ids = preprocessor.corpus_to_bow(corpus)
    embeddings        = np.array([vocab_embs[w] for w in vocab])

    print("Caching BoW data...")
    np.save(cache['bow_data'],       bow_data)
    np.save(cache['word_embeddings'], embeddings)
    vocab_data = {'vocab': vocab, 'word_to_idx': preprocessor.word_to_idx}
    with open(cache['vocab'], 'wb') as f:
        pickle.dump(vocab_data, f)
    with open(cache['doc_ids'], 'wb') as f:
        pickle.dump(doc_ids, f)

    print(f"BoW ready: {bow_data.shape}, vocab: {len(vocab)}, embs: {embeddings.shape}")
    return bow_data, vocab, embeddings, doc_ids


if __name__ == "__main__":
    from dataset_loader import load_dataset
    corpus, _, _ = load_dataset("msmarco")
    bow_data, vocab, embeddings, doc_ids = prepare_bow_data(corpus, "msmarco")
    print(f"BoW shape       : {bow_data.shape}")
    print(f"Vocabulary size : {len(vocab)}")
    print(f"Embedding dim   : {embeddings.shape[1]}")
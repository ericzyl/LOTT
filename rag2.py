import time
import numpy as np
from data import loader
import hott
import lot
from bert import create_bert_embeddings, bow_to_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from tqdm import tqdm

DATA_PATH = "./data/bbcsport-emd_tr_te_split.mat"
EMBED_PATH = "./data/glove.6B/glove.6B.300d.txt"

seed = 42

class SBERTEmbeddingWrapper(Embeddings):
    def __init__(self, vocab):
        self.vocab = vocab
    
    def embed_documents(self, texts):
        # Converting Texts to BoW format and Embedding
        bow_representations = []
        for text in texts:
            words = text.split()
            bow = np.zeros(len(self.vocab))
            for word in words:
                if word in self.vocab:
                    idx = self.vocab.index(word)
                    bow[idx] += 1
            bow_representations.append(bow)
        return create_bert_embeddings(np.array(bow_representations), self.vocab)
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]


class CustomLOTEmbeddingWrapper(Embeddings):
    def __init__(self, vocab, gaussian_vector, lda_centers, cost_T, topic_model=None):
        self.vocab = vocab
        self.gaussian_vector = gaussian_vector
        self.lda_centers = lda_centers
        self.cost_T = cost_T
        self.topic_model = topic_model
    
    def embed_documents(self, texts):
        # Converting Texts to Topic Proportions and then to LOT Embeddings
        # Need actual Topic Model and Implementation here
        topic_proportions = self._texts_to_topics(texts)
        return lot.create_lot_embeddings(
            topic_proportions, 
            self.gaussian_vector, 
            self.lda_centers, 
            self.cost_T
        )
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]
    
    def _texts_to_topics(self, texts):
        # Needs to be implemented based on the Topic Model used
        raise NotImplementedError("Implement topic extraction from texts")

print("Loading data...")
data = loader(DATA_PATH, EMBED_PATH, p=1)
cost_E = data['cost_E']
cost_T = data['cost_T']
vocab = data["vocab"]
bow_data, y = data['X'], data['y']
topic_proportions = data['proportions']
lda_centers = data['lda_C']
embeddings = data['embeddings']

bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(
    bow_data, topic_proportions, y, random_state=seed
)

train_texts = bow_to_text(bow_train, vocab)
test_texts = bow_to_text(bow_test, vocab)

print(f"Train samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")

print("\n" + "="*80)
print("RAG PIPELINE 1: SBERT Embedding-based Retrieval")

print("Creating SBERT Embeddings for training data...")
start_time = time.time()
sbert_train_embeddings = create_bert_embeddings(bow_train, vocab)
sbert_embed_time = time.time() - start_time
print(f"Embedding creation time: {sbert_embed_time:.2f} seconds")

# Building FAISS Vector Store
print("Building FAISS Index for SBERT...")
build_start = time.time()

# Creating Documents with Metadata
sbert_documents = [
    Document(page_content=text, metadata={"label": int(label), "index": i}) 
    for i, (text, label) in enumerate(zip(train_texts, y_train))
]

# Creating Text-Embedding Pairs
sbert_text_embeddings = [(doc.page_content, emb) for doc, emb in zip(sbert_documents, sbert_train_embeddings)]

# Creating Embedding Wrapper
sbert_wrapper = SBERTEmbeddingWrapper(vocab)

# Building FAISS Index with Metadata
sbert_vectorstore = FAISS.from_embeddings(
    sbert_text_embeddings,
    sbert_wrapper,
    metadatas=[doc.metadata for doc in sbert_documents]
)

build_time = time.time() - build_start
print(f"FAISS index build time: {build_time:.2f} seconds")

# Performing Retrieval on Test Data
print("Performing retrieval on test data...")
retrieval_start = time.time()

sbert_test_embeddings = create_bert_embeddings(bow_test, vocab)

retrieved_labels_sbert = []
for i, (test_text, test_emb) in enumerate(tqdm(
    zip(test_texts, sbert_test_embeddings), 
    total=len(test_texts),
    desc="SBERT Retrieval"
)):
    # Using FAISS Similarity search with Pre-computed Embedding
    results = sbert_vectorstore.similarity_search_by_vector(test_emb, k=1)
    
    # Extracting Predicted Label from Metadata
    predicted_label = results[0].metadata['label']
    retrieved_labels_sbert.append(predicted_label)

retrieval_time = time.time() - retrieval_start

sbert_accuracy = accuracy_score(y_test, retrieved_labels_sbert)
total_sbert_time = sbert_embed_time + build_time + retrieval_time

print(f"\n SBERT RAG Results:")
print(f"Accuracy: {sbert_accuracy:.4f}")
print(f"Total Runtime: {total_sbert_time:.2f} seconds")
print(f"Embedding time: {sbert_embed_time:.2f}s")
print(f"Index build time: {build_time:.2f}s")
print(f"Retrieval time: {retrieval_time:.2f}s")

print("\n RAG PIPELINE 2: Custom LOT Embedding-based Retrieval")

print("Creating LOT Embeddings for training data...")
start_time = time.time()

gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhm=25)
custom_train_embeddings = lot.create_lot_embeddings(
    topic_train, 
    gaussian_vector, 
    lda_centers, 
    cost_T
)

custom_embed_time = time.time() - start_time
print(f"Embedding creation time: {custom_embed_time:.2f} seconds")

print("Building FAISS index for LOT...")
build_start = time.time()

# Reusing same Documents with Metadata
custom_documents = [
    Document(page_content=text, metadata={"label": int(label), "index": i}) 
    for i, (text, label) in enumerate(zip(train_texts, y_train))
]

# Create text-embedding pairs
custom_text_embeddings = [(doc.page_content, emb) for doc, emb in zip(custom_documents, custom_train_embeddings)]

# Create embedding wrapper (simplified - won't actually be used for querying with vectors)
custom_wrapper = CustomEmbeddingWrapper(lambda texts: custom_train_embeddings)

# Build FAISS index with metadata
custom_vectorstore = FAISS.from_embeddings(
    custom_text_embeddings,
    custom_wrapper,
    metadatas=[doc.metadata for doc in custom_documents]
)

build_time = time.time() - build_start
print(f"FAISS index build time: {build_time:.2f} seconds")

print("Performing retrieval on test data...")
retrieval_start = time.time()

custom_test_embeddings = lot.create_lot_embeddings(
    topic_test, 
    gaussian_vector, 
    lda_centers, 
    cost_T
)

retrieved_labels_custom = []
for i, (test_text, test_emb) in enumerate(tqdm(
    zip(test_texts, custom_test_embeddings), 
    total=len(test_texts),
    desc="LOT Retrieval"
)):
    results = custom_vectorstore.similarity_search_by_vector(test_emb, k=1)
    
    predicted_label = results[0].metadata['label']
    retrieved_labels_custom.append(predicted_label)

retrieval_time = time.time() - retrieval_start

custom_accuracy = accuracy_score(y_test, retrieved_labels_custom)
total_custom_time = custom_embed_time + build_time + retrieval_time

print(f"\n LOT RAG Results:")
print(f"Accuracy: {custom_accuracy:.4f}")
print(f"Total Runtime: {total_custom_time:.2f} seconds")
print(f"Embedding time: {custom_embed_time:.2f}s")
print(f"Index build time: {build_time:.2f}s")
print(f"Retrieval time: {retrieval_time:.2f}s")

print("\n COMPARISON")
print(f"{'Method':<20} {'Accuracy':<15} {'Total Time (s)':<15} {'Speedup':<10}")
print(f"{'SBERT':<20} {sbert_accuracy:<15.4f} {total_sbert_time:<15.2f} {'1.00x':<10}")
print(f"{'Custom LOT':<20} {custom_accuracy:<15.4f} {total_custom_time:<15.2f} {f'{total_sbert_time/total_custom_time:.2f}x':<10}")

if custom_accuracy > sbert_accuracy:
    print(f"Custom LOT is {(custom_accuracy - sbert_accuracy)*100:.2f}% more accurate")
elif sbert_accuracy > custom_accuracy:
    print(f"SBERT is {(sbert_accuracy - custom_accuracy)*100:.2f}% more accurate")
else:
    print("Both methods have equal accuracy")

if total_custom_time < total_sbert_time:
    print(f"Custom LOT is {total_sbert_time/total_custom_time:.2f}x faster")
elif total_sbert_time < total_custom_time:
    print(f"SBERT is {total_custom_time/total_sbert_time:.2f}x faster")
else:
    print("Both methods have equal runtime")
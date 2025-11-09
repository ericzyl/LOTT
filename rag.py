import time
import numpy as np
from data import loader
import hott
import lot
from bert import create_bert_embeddings, bow_to_text
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_core.documents import Document
from tqdm import tqdm

DATA_PATH = "./data/bbcsport-emd_tr_te_split.mat"
EMBED_PATH = "./data/glove.6B/glove.6B.300d.txt"

seed = 42

# Helper Class to wrap Embedding Generation for LangChain
class CustomEmbeddingWrapper(Embeddings):
    def __init__(self, embedding_func):
        self.embedding_func = embedding_func

    def embed_documents(self, texts):
        return self.embedding_func(texts)

    def embed_query(self, text):
        return self.embedding_func([text])[0]

# Loading Data by converting BoW to Text
data = loader(DATA_PATH, EMBED_PATH, p=1)
cost_E = data['cost_E']
cost_T = data['cost_T']
vocab = data["vocab"]
bow_data, y = data['X'], data['y']
topic_proportions = data['proportions']
lda_centers = data['lda_C']
embeddings  = data['embeddings']

bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(bow_data, topic_proportions, y, random_state=seed)

train_texts = bow_to_text(bow_train, vocab)
test_texts = bow_to_text(bow_test, vocab)

print("\n SBERT Embedding-based Retrieval")
start_time = time.time()
sbert_embeddings = create_bert_embeddings(bow_train, vocab)
X_test_sbert = create_bert_embeddings(bow_test, vocab)
sbert_time = time.time() - start_time

# Building FAISS index for SBERT using plain Text strings, not Documents
bert_text_embeddings = list(zip(train_texts, sbert_embeddings))

# Creating a Dummy Embedding Wrapper
dummy_sbert_wrapper = CustomEmbeddingWrapper(lambda texts: create_bert_embeddings(bow_train, vocab))

sbert_vectorstore = FAISS.from_embeddings(bert_text_embeddings, dummy_sbert_wrapper)

# Evaluating Retrieval Accuracy
retrieved_labels = []
for i, test_vec in enumerate(tqdm(X_test_sbert, desc="Evaluating SBERT Retrieval")):
    sims = cosine_similarity([test_vec], sbert_embeddings)[0]
    nearest_idx = np.argmax(sims)
    retrieved_labels.append(y_train[nearest_idx])

sbert_accuracy = accuracy_score(y_test, retrieved_labels)
print(f"SBERT Accuracy: {sbert_accuracy:.4f}")
print(f"SBERT Runtime: {sbert_time:.2f} seconds")

print("\n Custom Embedding-based Retrieval")
start_time = time.time()
gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhm=25)
custom_train_embs = lot.create_lot_embeddings(topic_train, gaussian_vector, lda_centers, cost_T)
custom_test_embs = lot.create_lot_embeddings(topic_test, gaussian_vector, lda_centers, cost_T)
custom_time = time.time() - start_time

# Building FAISS index for custom embeddings using Plain Text strings
custom_text_embeddings = list(zip(train_texts, custom_train_embs))

# Creating Wrapper for Custom Embeddings
custom_wrapper = CustomEmbeddingWrapper(lambda texts: custom_train_embs)

custom_vectorstore = FAISS.from_embeddings(custom_text_embeddings, custom_wrapper)

# Evaluating Retrieval Accuracy
retrieved_labels_custom = []
for i, test_vec in enumerate(tqdm(custom_test_embs, desc="Evaluating Custom Retrieval")):
    sims = cosine_similarity([test_vec], custom_train_embs)[0]
    nearest_idx = np.argmax(sims)
    retrieved_labels_custom.append(y_train[nearest_idx])

custom_accuracy = accuracy_score(y_test, retrieved_labels_custom)
print(f"Custom Accuracy: {custom_accuracy:.4f}")
print(f"Custom Runtime: {custom_time:.2f} seconds")
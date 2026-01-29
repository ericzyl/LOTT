# Copied and Improved from rag.py
# In rag.py, we weren't using FAISS for retrieval efficiently and just using simple Cosine Similarity that we don't want
# Also, the Embedding Wrappers weren't correct as each Input text returned the same fixed array of training embeddings- 
# It worked because FAISS wasn't used: but if FAISS was used, then the entire system would break

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

# Storing Train Embeddings in a Dictionary indexed by Text for FAISS so that we don't have to wastefully compute it each time
sbert_dict = {train_texts[i]: sbert_embeddings[i] for i in range(len(train_texts))}

X_test_sbert = create_bert_embeddings(bow_test, vocab)
sbert_time = time.time() - start_time

# Building FAISS index for SBERT using plain Text strings, not Documents
# Both BERT and Custom Text embeddings take 2 args here as FAISS.from_embeddings need only 2: 
# So, Metadata added later after processing FAISS.from_embeddings
bert_text_embeddings = [
    (train_texts[i], sbert_embeddings[i])
    for i in range(len(train_texts))
]

# Creating a Dummy Embedding Wrapper
sbert_wrapper = CustomEmbeddingWrapper(lambda texts: [sbert_dict[t] for t in texts])
sbert_wrapper.embed_query = lambda text: sbert_dict[text]

sbert_vectorstore = FAISS.from_embeddings(bert_text_embeddings, sbert_wrapper)
for i, doc in enumerate(sbert_vectorstore.docstore._dict.values()):
    doc.metadata = {"label": int(y_train[i])}

# Evaluating Retrieval Accuracy
retrieved_labels = []
for i, test_vec in enumerate(tqdm(X_test_sbert, desc="FAISS SBERT Retrieval")):
    result = sbert_vectorstore.similarity_search_by_vector(test_vec, k=1)
    retrieved_labels.append(result[0].metadata["label"])

sbert_accuracy = accuracy_score(y_test, retrieved_labels)
print(f"SBERT Accuracy: {sbert_accuracy:.4f}")
print(f"SBERT Runtime: {sbert_time:.2f} seconds")

print("\n Custom Embedding-based Retrieval")
start_time = time.time()
gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhm=25)
custom_train_embs = lot.create_lot_embeddings(topic_train, gaussian_vector, lda_centers, cost_T)
custom_test_embs = lot.create_lot_embeddings(topic_test, gaussian_vector, lda_centers, cost_T)
custom_dict = {train_texts[i]: custom_train_embs[i] for i in range(len(train_texts))}
custom_time = time.time() - start_time

# Building FAISS index for custom embeddings using Plain Text strings
custom_text_embeddings = [
    (train_texts[i], custom_train_embs[i])
    for i in range(len(train_texts))
]

# Creating Wrapper for Custom Embeddings
custom_wrapper = CustomEmbeddingWrapper(lambda texts: [custom_dict[t] for t in texts])
custom_wrapper.embed_query = lambda text: custom_dict[text]

custom_vectorstore = FAISS.from_embeddings(custom_text_embeddings, custom_wrapper)
for i, doc in enumerate(custom_vectorstore.docstore._dict.values()):
    doc.metadata = {"label": int(y_train[i])}

# Evaluating Retrieval Accuracy
retrieved_labels_custom = []
for i, test_vec in enumerate(tqdm(custom_test_embs, desc="FAISS Custom Retrieval")):
    result = custom_vectorstore.similarity_search_by_vector(test_vec, k=1)
    retrieved_labels_custom.append(result[0].metadata["label"])

custom_accuracy = accuracy_score(y_test, retrieved_labels_custom)
print(f"Custom Accuracy: {custom_accuracy:.4f}")
print(f"Custom Runtime: {custom_time:.2f} seconds")
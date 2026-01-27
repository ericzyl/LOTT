from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import time
import numpy as np

from sklearn.decomposition import PCA
from data import loader
from knn_classifier import knn

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import distances
import hott
import lot

import bert


data_path = './data/'

embeddings_path = './data/glove.6B/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
# data_name = 'bbcsport-emd_tr_te_split.mat'
# data_name = 'twitter-emd_tr_te_split.mat'
# data_name = 'r8-emd_tr_te3.mat'
# data_name = 'amazon-emd_tr_te_split.mat'
data_name = 'classic-emd_tr_te_split.mat'
# data_name = 'ohsumed-emd_tr_te_ix.mat'
# data_name = '20ng2_500-emd_tr_te.mat'
# data_name = 'recipe2-emd_tr_te_split.mat'

# # p=1 for W1 and p=2 for W2
# p = 1
# data = loader(data_path + data_name, embeddings_path, p=p)
# cost_E = data['cost_E']
# cost_T = data['cost_T']

# bow_data, y = data['X'], data['y']
# topic_proportions = data['proportions']
# lda_centers = data['lda_C']
# embeddings  = data['embeddings']

# seed = 0
# bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(bow_data, topic_proportions, y, random_state=seed)

# methods = {'LOTT': lot.lot,
#            'HOTT': hott.hott}
# methods = {'HOTT': hott.hott}
# methods = {'LOTT': lot.lot}
methods = {}

for method in methods.keys():
    if method in ['LOTT']:
        size = 70
        num_modal = 10
        fwhms = [10] * num_modal
        #centers = np.linspace(0, size-1, num_modal, dtype=int)
        if num_modal == 1:
            centers = [size // 2]  # Single center at the middle of the vector
        else:
            centers = [(i + 1) * size // (num_modal + 1) for i in range(num_modal)]
        gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhms=fwhms, centers=centers)
        X_train_lot = lot.create_lot_embeddings(topic_train, gaussian_vector, lda_centers, cost_T)
        X_test_lot = lot.create_lot_embeddings(topic_test, gaussian_vector, lda_centers, cost_T)

        X_combined = np.vstack([X_train_lot, X_test_lot])

        # Combine train and test labels
        y_combined = np.hstack([y_train, y_test])
        np.savez_compressed("lot_embeddings.npz", X_combined=X_combined, y_combined=y_combined)

        # Apply t-SNE to reduce the combined LOT embeddings to 2D
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X_combined)

        # Define unique labels and colors
        unique_labels = np.unique(y_combined)
        # colors = ['red', 'green', 'blue', 'orange']  # Customize colors as needed
        colors = ['#cc398d', '#aba7ce', '#86bedc', '#fd9f5a']
        label_names = ['CACM', 'MED', 'CRAN', 'CISI']  # Customize names as needed

        # Plot the t-SNE results with class labels
        plt.figure(figsize=(6, 6))

        # Plot each class with a specific color and label
        for i, label in enumerate(unique_labels):
            plt.scatter(X_tsne[y_combined == label, 0], X_tsne[y_combined == label, 1], 
                        color=colors[i], label=label_names[i], s=1, alpha=0.7)

        # Add a legend
        plt.legend(loc="upper left", fontsize=15)

        # Title and axis labels
        plt.xticks([])
        plt.yticks([])

        # Save the plot as a high-resolution image
        plt.savefig("tsne_lot_embeddings_with.png", dpi=300, bbox_inches='tight')
        #plt.show()

        # Compute test error
        # test_error = knn(X_train_lot, X_test_lot, y_train, y_test, methods[method], C, n_neighbors=7)
        knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
        knn_classifier.fit(X_train_lot, y_train)
        y_pred = knn_classifier.predict(X_test_lot)
        test_error = 1 - accuracy_score(y_test, y_pred)
    else:
        # If method is HOTT or HOFTT train LDA and compute topic-topic transport cost
        # X_train, X_test = topic_train, topic_test
        X_train, X_test = bow_train, bow_test

        # Apply t-SNE with the specified metric
        X_combined2 = np.vstack([X_train, X_test])
        y_combined2 = np.hstack([y_train, y_test])
        np.savez_compressed("embeddings3.npz", X_combined=X_combined2, y_combined=y_combined2)

        # tsne = TSNE(n_components=2, random_state=42)
        # X_tsne2 = tsne.fit_transform(X_combined2)

        # unique_labels = np.unique(y_combined2)
        # colors = ['red', 'green', 'blue', 'orange']  # Customize colors as needed
        # label_names = ['CACM', 'MED', 'CRAN', 'CISI']  # Customize names as needed

        # # Plot the t-SNE results with class labels
        # plt.figure(figsize=(6, 6))

        # # Plot each class with a specific color and label
        # for i, label in enumerate(unique_labels):
        #     plt.scatter(X_tsne2[y_combined2 == label, 0], X_tsne2[y_combined2 == label, 1], 
        #                 color=colors[i], label=label_names[i], s=50, alpha=0.7)

        # # Add a legend
        # plt.legend(loc="upper left", fontsize=15)

        # # Title and axis labels
        # plt.xticks([])
        # plt.yticks([])

        # # Save the plot as a high-resolution image
        # plt.savefig("tsne_embeddings.png", dpi=300, bbox_inches='tight')

    print(f"{method} test error: {test_error:.4f}")

# ============================================================================
# BERT Embeddings Visualization (using raw text)
# ============================================================================

import re

# Specify which BERT model to visualize
bert_model_name = 'SBERT'  # Choose from: SBERT, SBERT-large, DistilBERT, RoBERTa, BERT

print(f"\n{'='*80}")
print(f"Generating t-SNE plot for {bert_model_name}")
print(f"{'='*80}")

# Load raw text dataset
raw_data_file = './Raw_Datasets/all_classic.txt'  # Change to match your dataset

def load_raw_text(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            label, text = parts
            label = label.strip('"')
            text = re.sub(r'\s+', ' ', text).strip()
            texts.append(text)
            labels.append(label)
    
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    labels_int = np.array([label_to_idx[label] for label in labels])
    return texts, labels_int

# Load raw texts
texts, labels = load_raw_text(raw_data_file)

# Split using same seed as original
from sklearn.model_selection import train_test_split
X_train_text, X_test_text, y_train_bert, y_test_bert = train_test_split(
    texts, labels, test_size=0.2, random_state=0, stratify=labels
)

# Create BERT embeddings
print(f"Creating {bert_model_name} embeddings...")
embedder = bert.BERTDocumentEmbedder(
    model_name=bert.BERT_MODELS[bert_model_name],
    aggregation='mean'
)

X_train_bert = embedder.encode_documents(X_train_text, batch_size=16)
X_test_bert = embedder.encode_documents(X_test_text, batch_size=16)

# Combine train and test
X_combined_bert = np.vstack([X_train_bert, X_test_bert])
y_combined_bert = np.hstack([y_train_bert, y_test_bert])

# Save BERT embeddings
np.savez_compressed(f"{bert_model_name.lower()}_embeddings.npz", 
                    X_combined=X_combined_bert, 
                    y_combined=y_combined_bert)

# Apply t-SNE
print("Applying t-SNE...")
tsne_bert = TSNE(n_components=2, random_state=42)
X_tsne_bert = tsne_bert.fit_transform(X_combined_bert)

# Plot
unique_labels = np.unique(y_combined_bert)
colors = ['#cc398d', '#aba7ce', '#86bedc', '#fd9f5a']
label_names = ['CACM', 'MED', 'CRAN', 'CISI']

plt.figure(figsize=(6, 6))

for i, label in enumerate(unique_labels):
    plt.scatter(X_tsne_bert[y_combined_bert == label, 0], 
                X_tsne_bert[y_combined_bert == label, 1], 
                color=colors[i], label=label_names[i], s=1, alpha=0.7)

plt.legend(loc="upper left", fontsize=15)
plt.xticks([])
plt.yticks([])

plt.savefig(f"tsne_{bert_model_name.lower()}_embeddings.png", dpi=300, bbox_inches='tight')
print(f"Saved plot to: tsne_{bert_model_name.lower()}_embeddings.png")

# Compute test error for BERT
knn_bert = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
knn_bert.fit(X_train_bert, y_train_bert)
y_pred_bert = knn_bert.predict(X_test_bert)
test_error_bert = 1 - accuracy_score(y_test_bert, y_pred_bert)

print(f"{bert_model_name} test error: {test_error_bert:.4f}")
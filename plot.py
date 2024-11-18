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

# p=1 for W1 and p=2 for W2
p = 1
data = loader(data_path + data_name, embeddings_path, p=p)
cost_E = data['cost_E']
cost_T = data['cost_T']

bow_data, y = data['X'], data['y']
topic_proportions = data['proportions']
lda_centers = data['lda_C']
embeddings  = data['embeddings']

seed = 0
bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(bow_data, topic_proportions, y, random_state=seed)

# methods = {'LOTT': lot.lot,
#            'HOTT': hott.hott}
methods = {'HOTT': hott.hott}

for method in methods.keys():
    if method in ['LOTT']:
        gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhm=25)
        X_train_lot = lot.create_lot_embeddings(topic_train, gaussian_vector, lda_centers, cost_T)
        X_test_lot = lot.create_lot_embeddings(topic_test, gaussian_vector, lda_centers, cost_T)

        X_combined = np.vstack([X_train_lot, X_test_lot])

        # Combine train and test labels
        y_combined = np.hstack([y_train, y_test])
        np.savez_compressed("lot_embeddings.npz", X_combined=X_combined, y_combined=y_combined)

        # Apply t-SNE to reduce the combined LOT embeddings to 2D
        # tsne = TSNE(n_components=2, random_state=42)
        # X_tsne = tsne.fit_transform(X_combined)

        # # Define unique labels and colors
        # unique_labels = np.unique(y_combined)
        # colors = ['red', 'green', 'blue', 'orange']  # Customize colors as needed
        # label_names = ['CACM', 'MED', 'CRAN', 'CISI']  # Customize names as needed

        # # Plot the t-SNE results with class labels
        # plt.figure(figsize=(6, 6))

        # # Plot each class with a specific color and label
        # for i, label in enumerate(unique_labels):
        #     plt.scatter(X_tsne[y_combined == label, 0], X_tsne[y_combined == label, 1], 
        #                 color=colors[i], label=label_names[i], s=50, alpha=0.7)

        # # Add a legend
        # plt.legend(loc="upper left", fontsize=15)

        # # Title and axis labels
        # plt.xticks([])
        # plt.yticks([])

        # # Save the plot as a high-resolution image
        # plt.savefig("tsne_lot_embeddings_with.png", dpi=300, bbox_inches='tight')
        # #plt.show()

        
        # Compute test error
        # test_error = knn(X_train, X_test, y_train, y_test, methods[method], C, n_neighbors=7)
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


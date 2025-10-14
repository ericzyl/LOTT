from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
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

# NLTK demands Stopwords to be downloaded first now
import nltk
nltk.download('stopwords')

# Download datasets used by Kusner et al from
# https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# and put them into
data_path = './data/'


# Download GloVe 6B tokens, 300d word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './data/glove.6B/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
# data_name = 'bbcsport-emd_tr_te_split.mat'
data_name = 'twitter-emd_tr_te_split.mat'
# data_name = 'r8-emd_tr_te3.mat'
# data_name = 'amazon-emd_tr_te_split.mat'
# data_name = 'classic-emd_tr_te_split.mat'
# data_name = 'ohsumed-emd_tr_te_ix.mat'

# data_name = '20ng2_500-emd_tr_te.mat'
# data_name = 'recipe2-emd_tr_te_split.mat'

# p=1 for W1 and p=2 for W2
p = 1
# For Standard GloVe Embeddings
# data = loader(data_path + data_name, embeddings_path, p=p)
# For BERT Embeddings
data = loader(data_path + data_name, embeddings_path, p=p,
              glove_embeddings=False, bert_model_name='roberta-large')
# Getting the following Test errors on Twitter Dataset and Mac M4 CPU-
# With GloVe, LOTT- 0.32175; With BERT-Base-Uncased, LOTT- 0.342342; with DistilBERT-Base-Uncased, LOTT- 0.33333; 
# With RoBERTa-Base, LOTT- 0.35512; with BERT-Large-Uncased, LOTT-0.324324; with RoBERTa-Large, LOTT- 0.350064;

cost_E = data['cost_E']
cost_T = data['cost_T']

bow_data, y = data['X'], data['y']
topic_proportions = data['proportions']
lda_centers = data['lda_C']
embeddings  = data['embeddings']

seed = 0
bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(bow_data, topic_proportions, y, random_state=seed)

# ###################
# classes_to_keep = [1, 2,3]

# # Create a mask to filter out only the specified classes in the training set
# train_mask = np.isin(y_train, classes_to_keep)
# test_mask = np.isin(y_test, classes_to_keep)

# # Apply the mask to truncate the training and test sets
# topic_train = topic_train[train_mask]
# y_train = y_train[train_mask]
# topic_test = topic_test[test_mask]
# y_test = y_test[test_mask]
# #####################

# Pick a method among RWMD, WMD, WMD-T20, HOTT, HOFTT
methods = {'LOTT': lot.lot,
           'HOTT': hott.hott,
           'HOFTT': hott.hoftt,
           'WMD-T20': lambda p, q, C: distances.wmd(p, q, C, truncate=20),
           'RWMD': distances.rwmd,
           'WMD': distances.wmd}
    
for method in methods.keys():
    
    t_s = time.time()
    # Get train/test data representation and transport cost
    if method in ['LOTT']:
        # create lot embedding
        #bimodal_vector = lot.makeBimodal1D(size=topic_train.shape[1], fwhm1=10, center1=20, fwhm2=5, center2=80)
        gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhm=25)
        X_train_lot = lot.create_lot_embeddings(topic_train, gaussian_vector, lda_centers, cost_T)
        X_test_lot = lot.create_lot_embeddings(topic_test, gaussian_vector, lda_centers, cost_T)

        # Replaced with BERT document wide embeddings

        # X_train, X_test = normalize(bow_train, 'l1'), normalize(bow_test, 'l1')

        # train knn classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
        knn_classifier.fit(X_train_lot, y_train)
        y_pred = knn_classifier.predict(X_test_lot)


        # Compute test error
        test_error = 1 - accuracy_score(y_test, y_pred)
        runtime = time.time() - t_s
        num_pairs = len(X_test_lot) * len(X_train_lot)
        print(len(X_test_lot))
        print(len(X_train_lot))
        pairs_per_second = num_pairs / runtime

    # if method in ['LOWT']:
    #     # create lot embedding
    #     #bimodal_vector = lot.makeBimodal1D(size=topic_train.shape[1], fwhm1=10, center1=20, fwhm2=5, center2=80)
    #     gaussian_vector = lot.makeGaussian1D(size=bow_train.shape[1], fwhm=25)
    #     X_train_lot = lot.lot_wmd_embeddings(bow_train, gaussian_vector, embeddings, cost_E)
    #     X_test_lot = lot.lot_wmd_embeddings(bow_test, gaussian_vector, embeddings, cost_E)

    #     # X_train, X_test = normalize(bow_train, 'l1'), normalize(bow_test, 'l1')

    #     # train knn classifier
    #     knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
    #     knn_classifier.fit(X_train_lot, y_train)
    #     y_pred = knn_classifier.predict(X_test_lot)


    #     # Compute test error
    #     test_error = 1 - accuracy_score(y_test, y_pred)

    else:
        if method in ['HOTT', 'HOFTT']:
            # If method is HOTT or HOFTT train LDA and compute topic-topic transport cost
            X_train, X_test = topic_train, topic_test
            C = data['cost_T']
            num_pairs = len(X_test) * len(X_train)
            
        else:
            # Normalize BOW and compute word-word transport cost
            X_train, X_test = normalize(bow_train, 'l1'), normalize(bow_test, 'l1')
            C = data['cost_E']
            num_pairs = len(X_test) * len(X_train)
        
        # Compute test error
        test_error = knn(X_train, X_test, y_train, y_test, methods[method], C, n_neighbors=7)
        runtime = time.time() - t_s
        pairs_per_second = num_pairs / runtime

    print(method + ' test error is %f; took %.2f seconds; %.2f of pairs per second' % (test_error, runtime, pairs_per_second))

# Done!

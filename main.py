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

import bert

# Download datasets used by Kusner et al from
# https://www.dropbox.com/sh/nf532hddgdt68ix/AABGLUiPRyXv6UL2YAcHmAFqa?dl=0
# and put them into
data_path = './data/'


# Download GloVe 6B tokens, 300d word embeddings from
# https://nlp.stanford.edu/projects/glove/
# and put them into
embeddings_path = './data/glove.6B/glove.6B.300d.txt'

# Pick a dataset (uncomment the line you want)
data_name = 'bbcsport-emd_tr_te_split.mat'
# data_name = 'twitter-emd_tr_te_split.mat'
# data_name = 'r8-emd_tr_te3.mat'
# data_name = 'amazon-emd_tr_te_split.mat'
# data_name = 'classic-emd_tr_te_split.mat'
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
methods = {
        'LOTT': lot.lot,
           'HOTT': hott.hott,
           'HOFTT': hott.hoftt,
           'WMD-T20': lambda p, q, C: distances.wmd(p, q, C, truncate=20),
           'RWMD': distances.rwmd,
           'WMD': distances.wmd,
           # BERT Methods
           'SBERT': None,
           'SBERT-large': None,
           'DistilBERT': None,
           'RoBERTa': None,
           'BERT': None
           }
    
vocab = data['vocab'] # Vocabulary obtained from Data Object

for method in methods.keys():
    
    t_s = time.time()
    
    if method in ['SBERT', 'SBERT-large', 'DistilBERT', 'RoBERTa', 'BERT']:
        print(f"\nProcessing {method}...")
        
        # Creating BERT embeddings for train and test
        model_name = bert.BERT_MODELS[method]
        X_train_bert = bert.create_bert_embeddings(
            bow_train, vocab, 
            model_name=model_name,
            aggregation='mean',
            batch_size=16 # Can reduce Batch Size if running out of Memory
        )
        X_test_bert = bert.create_bert_embeddings(
            bow_test, vocab,
            model_name=model_name,
            aggregation='mean',
            batch_size=16
        )
        
        # Training KNN Classifier
        knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
        knn_classifier.fit(X_train_bert, y_train)
        y_pred = knn_classifier.predict(X_test_bert)
        
        # Computing Test Error
        test_error = 1 - accuracy_score(y_test, y_pred)
        runtime = time.time() - t_s
        num_pairs = len(X_test_bert) * len(X_train_bert)
        pairs_per_second = num_pairs / runtime

        # Results: Test Errors corresponding to different BERT Embeddings for BBC Sport Dataset
        # SBERT- 0.140541, SBERT-large- 0.086486, DistilBERT- 0.081081, RoBERTa- 0.167568, BERT- 0.081081
        # Comparing with LDA Topic Methods-
        # LOTT- 0.140541, HOTT- 0.021622, HOFTT- 0.021622, WMD-T20- 0.043243, RWMD- 0.032432, WMD- 0.027027

    # Get train/test data representation and transport cost
    elif method in ['LOTT']:
        # create lot embedding
        #bimodal_vector = lot.makeBimodal1D(size=topic_train.shape[1], fwhm1=10, center1=20, fwhm2=5, center2=80)
        gaussian_vector = lot.makeGaussian1D(size=topic_train.shape[1], fwhm=25)
        X_train_lot = lot.create_lot_embeddings(topic_train, gaussian_vector, lda_centers, cost_T)
        X_test_lot = lot.create_lot_embeddings(topic_test, gaussian_vector, lda_centers, cost_T)

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
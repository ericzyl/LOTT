import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

from data import loader
import bert

data_path = '/kaggle/input/bert-lott-exps-datasets/'
embeddings_path = '/kaggle/input/bert-lott-exps-datasets/glove.6B.300d.txt'

datasets = [
    'bbcsport-emd_tr_te_split.mat',
    'twitter-emd_tr_te_split.mat',
    'r8-emd_tr_te3.mat',
    'amazon-emd_tr_te_split.mat',
    'classic-emd_tr_te_split.mat',
    'ohsumed-emd_tr_te_ix.mat',
    '20ng2_500-emd_tr_te.mat',
    'recipe2-emd_tr_te_split.mat'
]

bert_models = ['SBERT', 'SBERT-large', 'DistilBERT', 'RoBERTa', 'BERT']

p = 1
seed = 0

for data_name in datasets:
    try:
        print(f"\n==============================")
        print(f"Dataset: {data_name}")
        print(f"==============================")

        data = loader(data_path + data_name, embeddings_path, p=p)
        bow_data, y = data['X'], data['y']
        vocab = data['vocab']
        topic_proportions = data['proportions']

        bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(
            bow_data, topic_proportions, y, random_state=seed
        )

        for method in bert_models:
            t_s = time.time()
            print(f"\nProcessing {method}...")

            model_name = bert.BERT_MODELS[method]

            X_train_bert = bert.create_bert_embeddings(
                bow_train, vocab,
                model_name=model_name,
                aggregation='mean',
                batch_size=16
            )
            X_test_bert = bert.create_bert_embeddings(
                bow_test, vocab,
                model_name=model_name,
                aggregation='mean',
                batch_size=16
            )

            knn_classifier = KNeighborsClassifier(n_neighbors=7, metric='euclidean')
            knn_classifier.fit(X_train_bert, y_train)
            y_pred = knn_classifier.predict(X_test_bert)

            test_error = 1 - accuracy_score(y_test, y_pred)
            runtime = time.time() - t_s
            num_pairs = len(X_test_bert) * len(X_train_bert)
            pairs_per_second = num_pairs / runtime

            print(f"{method} test error: {test_error:.6f}; "
                f"time: {runtime:.2f}s; "
                f"{pairs_per_second:.2f} pairs/s")

    except Exception as e:
        print(f"An error occurred while processing dataset {data_name}: {e}")

print("\nAll datasets processed successfully with all BERT models.")
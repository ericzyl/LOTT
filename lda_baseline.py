import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize

from data import loader

# Config

data_path       = './data/'
embeddings_path = './data/glove.6B/glove.6B.300d.txt'

datasets = {
    'bbcsport': 'bbcsport-emd_tr_te_split.mat',
    'twitter':  'twitter-emd_tr_te_split.mat',
    'r8':       'r8-emd_tr_te3.mat',
    'amazon':   'amazon-emd_tr_te_split.mat',
    'classic':  'classic-emd_tr_te_split.mat',
    'ohsumed':  'ohsumed-emd_tr_te_ix.mat',
}

K_lda       = 70
n_neighbors = 7
seed        = 0
p           = 1

# Three LDA embedding variants to compare

def run_lda_variants(topic_train, topic_test, y_train, y_test):
    results = {}

    # Variant 1: raw topic proportions, euclidean KNN
    t = time.time()
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    clf.fit(topic_train, y_train)
    err = 1 - accuracy_score(y_test, clf.predict(topic_test))
    results['LDA-euclidean'] = (err, time.time() - t)

    # Variant 2: L2 normalised, euclidean KNN
    # (equivalent to cosine similarity KNN)
    t = time.time()
    tr_norm = sk_normalize(topic_train, norm='l2')
    te_norm = sk_normalize(topic_test,  norm='l2')
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    clf.fit(tr_norm, y_train)
    err = 1 - accuracy_score(y_test, clf.predict(te_norm))
    results['LDA-cosine'] = (err, time.time() - t)

    # Variant 3: cosine distance KNN directly
    t = time.time()
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    clf.fit(topic_train, y_train)
    err = 1 - accuracy_score(y_test, clf.predict(topic_test))
    results['LDA-cosine-direct'] = (err, time.time() - t)

    return results


all_results = {}

for ds_name, ds_file in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name.upper()}")
    print(f"{'='*60}")

    try:
        data = loader(data_path + ds_file, embeddings_path, p=p, K_lda=K_lda)
    except Exception as e:
        print(f"  Could not load {ds_file}: {e}")
        continue

    bow_data         = data['X']
    y                = data['y']
    topic_proportions = data['proportions']

    bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(
        bow_data, topic_proportions, y, random_state=seed
    )

    print(f"  Train: {len(y_train)} docs | Test: {len(y_test)} docs | "
          f"Classes: {len(np.unique(y))}")

    results = run_lda_variants(topic_train, topic_test, y_train, y_test)
    all_results[ds_name] = results

    for method, (err, t) in results.items():
        print(f"  {method:<25} error={err:.6f}  time={t:.3f}s")

# Summary table

print(f"\n\n{'='*70}")
print("SUMMARY TABLE — LDA EMBEDDING BASELINE")
print(f"{'='*70}")

methods = ['LDA-euclidean', 'LDA-cosine', 'LDA-cosine-direct']
header  = f"{'Dataset':<12}" + "".join(f"{m:<22}" for m in methods)
print(header)
print("-" * 70)

for ds_name in datasets:
    if ds_name not in all_results:
        continue
    row = f"{ds_name:<12}"
    for m in methods:
        err, t = all_results[ds_name].get(m, (float('nan'), 0))
        row += f"{err:.6f} ({t:.2f}s)     "
    print(row)

print(f"{'='*70}")
print("\nNote: lower error = better")
print("These results use raw LDA topic proportions as document embeddings.")
print("No optimal transport is applied — this isolates the contribution")
print("of the OT step in LOTT over a pure LDA representation.")
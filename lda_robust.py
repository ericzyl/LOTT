import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize as sk_normalize

from data import loader

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

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

SEEDS       = [0, 1, 2, 3, 4, 5]
K_lda       = 70
n_neighbors = 7
p           = 1
OUTPUT_PATH = './lda_baseline_results.json'

# -------------------------------------------------------------------
# LDA variants runner
# -------------------------------------------------------------------

def run_lda_variants(topic_train, topic_test, y_train, y_test):
    results = {}

    # Variant 1: raw topic proportions, euclidean KNN
    t = time.time()
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    clf.fit(topic_train, y_train)
    err = 1 - accuracy_score(y_test, clf.predict(topic_test))
    results['LDA-euclidean'] = {'error': float(err), 'time': time.time() - t}

    # Variant 2: L2 normalised, euclidean KNN (equivalent to cosine KNN)
    t = time.time()
    tr_norm = sk_normalize(topic_train, norm='l2')
    te_norm = sk_normalize(topic_test,  norm='l2')
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    clf.fit(tr_norm, y_train)
    err = 1 - accuracy_score(y_test, clf.predict(te_norm))
    results['LDA-cosine'] = {'error': float(err), 'time': time.time() - t}

    # Variant 3: cosine distance KNN directly
    t = time.time()
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='cosine')
    clf.fit(topic_train, y_train)
    err = 1 - accuracy_score(y_test, clf.predict(topic_test))
    results['LDA-cosine-direct'] = {'error': float(err), 'time': time.time() - t}

    return results


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------

# Structure:
# all_results[dataset][method] = {
#     'runs': [ {seed, error, time}, ... ],
#     'mean_error': float,
#     'std_error':  float,
#     'mean_time':  float,
# }

all_results = {}
methods     = ['LDA-euclidean', 'LDA-cosine', 'LDA-cosine-direct']

for ds_name, ds_file in datasets.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name.upper()}")
    print(f"{'='*60}")

    try:
        data = loader(data_path + ds_file, embeddings_path, p=p, K_lda=K_lda)
    except Exception as e:
        print(f"  Could not load {ds_file}: {e}")
        continue

    bow_data          = data['X']
    y                 = data['y']
    topic_proportions = data['proportions']

    # initialise storage per method
    ds_results = {m: {'runs': []} for m in methods}

    for seed in SEEDS:
        print(f"\n  Seed {seed}")

        bow_train, bow_test, topic_train, topic_test, y_train, y_test = train_test_split(
            bow_data, topic_proportions, y, random_state=seed
        )

        run_results = run_lda_variants(topic_train, topic_test, y_train, y_test)

        for m in methods:
            err = run_results[m]['error']
            t   = run_results[m]['time']
            ds_results[m]['runs'].append({
                'seed':  seed,
                'error': err,
                'time':  t,
            })
            print(f"    {m:<25} error={err:.6f}  time={t:.3f}s")

    # compute mean and std across seeds for each method
    for m in methods:
        errors = [r['error'] for r in ds_results[m]['runs']]
        times  = [r['time']  for r in ds_results[m]['runs']]
        ds_results[m]['mean_error'] = float(np.mean(errors))
        ds_results[m]['std_error']  = float(np.std(errors))
        ds_results[m]['mean_time']  = float(np.mean(times))

    all_results[ds_name] = ds_results

    # print per-dataset summary
    print(f"\n  Summary across {len(SEEDS)} seeds:")
    print(f"  {'Method':<25} {'Mean Error':>12} {'Std':>8} {'Mean Time':>12}")
    print(f"  {'-'*60}")
    for m in methods:
        r = ds_results[m]
        print(f"  {m:<25} {r['mean_error']:>12.6f} "
              f"{r['std_error']:>8.6f} {r['mean_time']:>11.3f}s")


# -------------------------------------------------------------------
# Save results to JSON
# -------------------------------------------------------------------

import os
os.makedirs('./results', exist_ok=True)

with open(OUTPUT_PATH, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}")


# -------------------------------------------------------------------
# Final summary table printed to terminal
# -------------------------------------------------------------------

print(f"\n\n{'='*80}")
print("FINAL SUMMARY — LDA EMBEDDING BASELINE (mean ± std across 6 seeds)")
print(f"{'='*80}")

header = f"{'Dataset':<12}" + "".join(f"{m:<28}" for m in methods)
print(header)
print("-" * 80)

for ds_name in datasets:
    if ds_name not in all_results:
        continue
    row = f"{ds_name:<12}"
    for m in methods:
        r = all_results[ds_name][m]
        row += f"{r['mean_error']:.4f} ± {r['std_error']:.4f}          "
    print(row)

print(f"{'='*80}")
print("\nNote: lower error = better")
print(f"Full per-run results saved to {OUTPUT_PATH}")
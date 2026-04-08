import time
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize as sk_normalize

from data import loader
from knn_classifier import knn

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
OUTPUT_PATH = './lda_baseline_results_new.json'

# -------------------------------------------------------------------
# Distance functions (no OT — pure geometry on topic vectors)
# The custom knn() expects: method(doc, x, C) -> scalar distance.
# We ignore C entirely for these baselines.
# -------------------------------------------------------------------

def euclidean_dist(a, b, C=None):
    diff = a - b
    return np.sqrt(diff @ diff)

def cosine_dist(a, b, C=None):
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return 1.0 - (a @ b) / denom

def cosine_dist_normalised(a, b, C=None):
    """Cosine distance on L2-normalised vectors — equivalent to half squared
    Euclidean distance, but kept separate for clarity."""
    return cosine_dist(a, b, C)

# Map method names to (distance_fn, needs_normalisation)
METHODS = {
    'LDA-euclidean':      (euclidean_dist,          False),
    'LDA-cosine':         (cosine_dist,              False),
    'LDA-cosine-l2norm':  (cosine_dist_normalised,   True),   # pre-normalise vectors
}

# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------

def run_lda_variants(topic_train, topic_test, y_train, y_test):
    results = {}

    # C is unused by all distance functions above; pass None
    C = None

    for method_name, (dist_fn, normalise) in METHODS.items():
        t = time.time()

        if normalise:
            tr = sk_normalize(topic_train, norm='l2')
            te = sk_normalize(topic_test,  norm='l2')
        else:
            tr, te = topic_train, topic_test

        err = knn(tr, te, y_train, y_test, dist_fn, C, n_neighbors=n_neighbors)
        elapsed = time.time() - t

        results[method_name] = {'error': float(err), 'time': elapsed}

    return results


# -------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------

all_results = {}
methods     = list(METHODS.keys())

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
            ds_results[m]['runs'].append({'seed': seed, 'error': err, 'time': t})
            print(f"    {m:<28} error={err:.6f}  time={t:.3f}s")

    # Aggregate across seeds
    for m in methods:
        errors = [r['error'] for r in ds_results[m]['runs']]
        times  = [r['time']  for r in ds_results[m]['runs']]
        ds_results[m]['mean_error'] = float(np.mean(errors))
        ds_results[m]['std_error']  = float(np.std(errors))
        ds_results[m]['mean_time']  = float(np.mean(times))

    all_results[ds_name] = ds_results

    print(f"\n  Summary across {len(SEEDS)} seeds:")
    print(f"  {'Method':<28} {'Mean Error':>12} {'Std':>8} {'Mean Time':>12}")
    print(f"  {'-'*63}")
    for m in methods:
        r = ds_results[m]
        print(f"  {m:<28} {r['mean_error']:>12.6f} "
              f"{r['std_error']:>8.6f} {r['mean_time']:>11.3f}s")


# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------

import os
os.makedirs('./results', exist_ok=True)

with open(OUTPUT_PATH, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {OUTPUT_PATH}")


# -------------------------------------------------------------------
# Final summary table
# -------------------------------------------------------------------

print(f"\n\n{'='*80}")
print("FINAL SUMMARY — LDA BASELINE (mean ± std across 6 seeds)")
print(f"{'='*80}")

header = f"{'Dataset':<12}" + "".join(f"{m:<32}" for m in methods)
print(header)
print("-" * 80)

for ds_name in datasets:
    if ds_name not in all_results:
        continue
    row = f"{ds_name:<12}"
    for m in methods:
        r = all_results[ds_name][m]
        row += f"{r['mean_error']:.4f} ± {r['std_error']:.4f}              "
    print(row)

print(f"{'='*80}")
print("\nNote: lower error = better")
print(f"Full per-run results saved to {OUTPUT_PATH}")
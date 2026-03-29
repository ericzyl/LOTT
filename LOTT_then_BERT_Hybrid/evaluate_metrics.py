from collections import defaultdict
from typing import Dict, List

import numpy as np

import config


def precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / k if k else 0.0


def recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def ndcg_at_k(retrieved: List[str], rel_scores: Dict[str, float], k: int) -> float:
    dcg = sum(
        rel_scores.get(d, 0.0) / np.log2(i + 2)
        for i, d in enumerate(retrieved[:k])
    )
    ideal = sorted(rel_scores.values(), reverse=True)[:k]
    idcg  = sum(r / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def mrr(retrieved: List[str], relevant: set) -> float:
    for rank, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def mean_average_precision(
    all_retrieved: List[List[str]],
    all_relevant:  List[set],
) -> float:
    aps = []
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        if not relevant:
            continue
        hits, precisions = 0, []
        for i, d in enumerate(retrieved, 1):
            if d in relevant:
                hits += 1
                precisions.append(hits / i)
        if precisions:
            aps.append(np.mean(precisions))
    return float(np.mean(aps)) if aps else 0.0


def evaluate_retrieval(
    all_retrieved: List[List[str]],
    query_ids:     List[str],
    qrels:         Dict[str, Dict[str, float]],
    method_name:   str,
) -> Dict:
    print(f"\nEvaluating '{method_name}'...")
    metrics: Dict[str, List[float]] = defaultdict(list)

    valid_pairs = [
        (qid, retrieved)
        for qid, retrieved in zip(query_ids, all_retrieved)
        if qid in qrels
    ]

    for qid, retrieved in valid_pairs:
        rel_scores = qrels[qid]
        relevant   = set(rel_scores.keys())
        for k in config.TOP_K_VALUES:
            metrics[f'P@{k}'].append(precision_at_k(retrieved, relevant, k))
            metrics[f'R@{k}'].append(recall_at_k(retrieved, relevant, k))
            metrics[f'NDCG@{k}'].append(ndcg_at_k(retrieved, rel_scores, k))
        metrics['MRR'].append(mrr(retrieved, relevant))

    all_retrieved_valid = [r for _, r in valid_pairs]
    all_relevant_valid  = [set(qrels[qid].keys()) for qid, _ in valid_pairs]
    map_score = mean_average_precision(all_retrieved_valid, all_relevant_valid)

    results = {m: float(np.mean(v)) for m, v in metrics.items()}
    results['MAP']         = map_score
    results['num_queries'] = len(valid_pairs)
    results['method']      = method_name

    # Pretty-print
    print(f"\n  {method_name}  (queries evaluated: {len(valid_pairs)})")
    print("  " + "-" * 58)
    for k in config.TOP_K_VALUES:
        print(f"  P@{k:<3}: {results[f'P@{k}']:.4f}  "
              f"R@{k:<3}: {results[f'R@{k}']:.4f}  "
              f"NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
    print(f"  MRR  : {results['MRR']:.4f}   MAP: {results['MAP']:.4f}")
    print("  " + "-" * 58)

    return results
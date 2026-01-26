"""
Evaluation metrics for retrieval
"""
import numpy as np
from typing import List, Dict
from collections import defaultdict
import config


def compute_precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Compute Precision@K"""
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_retrieved / k if k > 0 else 0.0


def compute_recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    """Compute Recall@K"""
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def compute_ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain@K"""
    retrieved_k = retrieved[:k]
    
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_k, 1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / np.log2(i + 1)
    
    # IDCG (ideal DCG)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(retrieved: List[str], relevant: set) -> float:
    """Compute Mean Reciprocal Rank"""
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(all_retrieved: List[List[str]], 
                       query_ids: List[str],
                       qrels: Dict[str, Dict[str, float]],
                       method_name: str) -> Dict:
    """
    Evaluate retrieval results
    
    Args:
        all_retrieved: List of retrieved doc lists for each query
        query_ids: List of query IDs
        qrels: Relevance judgments {query_id: {doc_id: score}}
        method_name: Name of the method (for display)
    
    Returns:
        Dictionary of metrics
    """
    print(f"\nEvaluating {method_name}...")
    
    metrics = defaultdict(list)
    
    for query_id, retrieved in zip(query_ids, all_retrieved):
        if query_id not in qrels:
            continue
        
        relevance_scores = qrels[query_id]
        relevant_docs = set(relevance_scores.keys())
        
        # Compute metrics for each K
        for k in config.TOP_K_VALUES:
            metrics[f'P@{k}'].append(compute_precision_at_k(retrieved, relevant_docs, k))
            metrics[f'R@{k}'].append(compute_recall_at_k(retrieved, relevant_docs, k))
            metrics[f'NDCG@{k}'].append(compute_ndcg_at_k(retrieved, relevance_scores, k))
        
        # MRR
        metrics['MRR'].append(compute_mrr(retrieved, relevant_docs))
    
    # Average metrics
    results = {metric: np.mean(values) for metric, values in metrics.items()}
    results['num_queries'] = len([qid for qid in query_ids if qid in qrels])
    results['method'] = method_name
    
    # Print results
    print(f"\n{method_name} Results:")
    print("-" * 60)
    for k in config.TOP_K_VALUES:
        print(f"P@{k}: {results[f'P@{k}']:.4f}  "
              f"R@{k}: {results[f'R@{k}']:.4f}  "
              f"NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
    print(f"MRR: {results['MRR']:.4f}")
    print(f"Queries evaluated: {results['num_queries']}")
    print("-" * 60)
    
    return results


if __name__ == "__main__":
    # Test metrics
    retrieved = ['doc1', 'doc2', 'doc3', 'doc4', 'doc5']
    relevant = {'doc1', 'doc3', 'doc6'}
    relevance_scores = {'doc1': 2, 'doc3': 1, 'doc6': 1}
    
    print(f"P@5: {compute_precision_at_k(retrieved, relevant, 5):.4f}")
    print(f"R@5: {compute_recall_at_k(retrieved, relevant, 5):.4f}")
    print(f"NDCG@5: {compute_ndcg_at_k(retrieved, relevance_scores, 5):.4f}")
    print(f"MRR: {compute_mrr(retrieved, relevant):.4f}")
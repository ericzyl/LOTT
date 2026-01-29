import numpy as np
import pickle
import time
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import config
from rag_bert import BERTRAGSystem, build_bert_rag_system
from rag_lott import LOTTRAGSystem, build_lott_rag_system, generate_lott_query_embeddings
from bert_embeddings import BERTEmbeddingGenerator


def compute_precision_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_retrieved / k if k > 0 else 0.0


def compute_recall_at_k(retrieved: List[str], relevant: set, k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    relevant_retrieved = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return relevant_retrieved / len(relevant)


def compute_mrr(retrieved: List[str], relevant: set) -> float:
    for i, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def compute_ndcg_at_k(retrieved: List[str], relevance_scores: Dict[str, float], k: int) -> float:
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


def compute_map(all_retrieved: List[List[str]], all_relevant: List[set]) -> float:
    average_precisions = []
    
    for retrieved, relevant in zip(all_retrieved, all_relevant):
        if len(relevant) == 0:
            continue
        
        precisions = []
        relevant_count = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                relevant_count += 1
                precision = relevant_count / i
                precisions.append(precision)
        
        if precisions:
            average_precisions.append(np.mean(precisions))
    
    return np.mean(average_precisions) if average_precisions else 0.0


class RAGEvaluator:
    
    def __init__(self, qrels: Dict[str, Dict[str, float]]):
        self.qrels = qrels
        self.results = {}
        
    def evaluate_system(self, system_name: str, all_retrieved: List[List[str]], 
                       query_ids: List[str], k_values: List[int] = None) -> Dict:

        if k_values is None:
            k_values = config.TOP_K_VALUES
        
        print(f"\nEvaluating {system_name} system...")
        
        metrics = defaultdict(list)
        
        for query_id, retrieved in zip(query_ids, all_retrieved):
            if query_id not in self.qrels:
                continue
            
            relevance_scores = self.qrels[query_id]
            relevant_docs = set(relevance_scores.keys())
            
            # Computing metrics for each K
            for k in k_values:
                metrics[f'P@{k}'].append(compute_precision_at_k(retrieved, relevant_docs, k))
                metrics[f'R@{k}'].append(compute_recall_at_k(retrieved, relevant_docs, k))
                metrics[f'NDCG@{k}'].append(compute_ndcg_at_k(retrieved, relevance_scores, k))
            
            # MRR
            metrics['MRR'].append(compute_mrr(retrieved, relevant_docs))
        
        # Computing MAP
        all_relevant = [set(self.qrels[qid].keys()) for qid in query_ids if qid in self.qrels]
        valid_retrieved = [ret for qid, ret in zip(query_ids, all_retrieved) if qid in self.qrels]
        metrics['MAP'] = [compute_map(valid_retrieved, all_relevant)]
        
        # Averaging metrics
        results = {metric: np.mean(values) for metric, values in metrics.items()}
        results['num_queries'] = len([qid for qid in query_ids if qid in self.qrels])
        
        self.results[system_name] = results
        
        return results
    
    def print_comparison(self):
        if len(self.results) < 2:
            print("Need at least 2 systems to compare")
            return
        
        print("\n" + "="*80)
        print("RETRIEVAL QUALITY COMPARISON")
        print("="*80)
        
        # Getting metric names
        metric_names = list(next(iter(self.results.values())).keys())
        metric_names.remove('num_queries')
        
        print(f"{'Metric':<15} {'BERT':<15} {'LOTT':<15} {'Difference':<15}")
        print("-" * 80)
        
        for metric in sorted(metric_names):
            bert_val = self.results['BERT'][metric]
            lott_val = self.results['LOTT'][metric]
            diff = lott_val - bert_val
            diff_pct = (diff / bert_val * 100) if bert_val > 0 else 0
            
            print(f"{metric:<15} {bert_val:<15.4f} {lott_val:<15.4f} "
                  f"{diff:+.4f} ({diff_pct:+.1f}%)")
        
        print("="*80)
    
    def plot_comparison(self, save_path=None):
        if len(self.results) < 2:
            print("Need at least 2 systems to compare")
            return
        
        # Extracting P@K, R@K, and NDCG@K metrics
        k_values = config.TOP_K_VALUES
        
        metrics_to_plot = [
            ('Precision@K', [f'P@{k}' for k in k_values]),
            ('Recall@K', [f'R@{k}' for k in k_values]),
            ('NDCG@K', [f'NDCG@{k}' for k in k_values])
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for ax, (title, metric_keys) in zip(axes, metrics_to_plot):
            bert_values = [self.results['BERT'][key] for key in metric_keys]
            lott_values = [self.results['LOTT'][key] for key in metric_keys]
            
            x = np.arange(len(k_values))
            width = 0.35
            
            ax.bar(x - width/2, bert_values, width, label='BERT', alpha=0.8)
            ax.bar(x + width/2, lott_values, width, label='LOTT', alpha=0.8)
            
            ax.set_xlabel('K')
            ax.set_ylabel('Score')
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(k_values)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = config.COMPARISON_PLOT_PATH
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComparison plot saved to {save_path}")
        plt.close()
    
    def save_results(self, save_path=None):
        if save_path is None:
            save_path = config.METRICS_RESULTS_PATH
        
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to {save_path}")


def benchmark_timing(bert_system: BERTRAGSystem, lott_system: LOTTRAGSystem,
                     bert_queries: np.ndarray, lott_queries: np.ndarray,
                     n_queries: int = 100) -> Dict:

    print("\n" + "="*80)
    print("BENCHMARKING RETRIEVAL TIME")
    print("="*80)
    
    # Limiting to n_queries for timing
    bert_q = bert_queries[:n_queries]
    lott_q = lott_queries[:n_queries]
    
    timing_results = {}
    
    # BERT timing
    print(f"\nTiming BERT retrieval ({n_queries} queries)...")
    start = time.time()
    bert_system.batch_search(bert_q, k=10, show_progress=False)
    bert_time = time.time() - start
    bert_per_query = bert_time / n_queries
    
    timing_results['BERT'] = {
        'total_time': bert_time,
        'per_query': bert_per_query,
        'queries_per_second': n_queries / bert_time
    }
    
    print(f"Total time: {bert_time:.2f}s")
    print(f"Per query: {bert_per_query*1000:.2f}ms")
    print(f"Throughput: {timing_results['BERT']['queries_per_second']:.1f} queries/sec")
    
    # LOTT timing
    print(f"\nTiming LOTT retrieval ({n_queries} queries)...")
    start = time.time()
    lott_system.batch_search(lott_q, k=10, show_progress=False)
    lott_time = time.time() - start
    lott_per_query = lott_time / n_queries
    
    timing_results['LOTT'] = {
        'total_time': lott_time,
        'per_query': lott_per_query,
        'queries_per_second': n_queries / lott_time
    }
    
    print(f"Total time: {lott_time:.2f}s")
    print(f"Per query: {lott_per_query*1000:.2f}ms")
    print(f"Throughput: {timing_results['LOTT']['queries_per_second']:.1f} queries/sec")
    
    # Comparison
    speedup = bert_time / lott_time
    print(f"\n{'LOTT is' if speedup > 1 else 'BERT is'} {abs(speedup):.2f}x "
          f"{'faster' if speedup != 1 else 'same speed'}")
    
    # Saving timing results
    with open(config.TIMING_RESULTS_PATH, 'w') as f:
        json.dump(timing_results, f, indent=2)
    
    return timing_results


def run_full_evaluation():
    """Run complete evaluation of both RAG systems"""
    print("\n" + "="*80)
    print("RUNNING FULL RAG EVALUATION")
    print("="*80)
    
    # Loading queries and qrels
    print("\nLoading queries and relevance judgments...")
    with open(config.QUERIES_PATH, 'rb') as f:
        queries = pickle.load(f)
    
    with open(config.QRELS_PATH, 'rb') as f:
        qrels = pickle.load(f)
    
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded qrels for {len(qrels)} queries")
    
    # Building/loading RAG systems
    bert_system = build_bert_rag_system()
    lott_system = build_lott_rag_system()
    
    # Loading document IDs for diagnostic
    with open(config.DATA_DIR / 'pipeline_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)
    doc_ids = metadata['doc_ids']
    
    # Generating query embeddings
    print("\n" + "="*80)
    print("GENERATING QUERY EMBEDDINGS")
    print("="*80)
    
    print("\nGenerating BERT query embeddings...")
    bert_generator = BERTEmbeddingGenerator()
    bert_query_embeddings, query_ids = bert_generator.generate_query_embeddings(queries)
    
    print("\nGenerating LOTT query embeddings...")
    lott_query_embeddings, lott_query_ids = generate_lott_query_embeddings(queries)
    
    # Performing retrieval
    print("\n" + "="*80)
    print("PERFORMING RETRIEVAL")
    print("="*80)
    
    print("\nBERT retrieval...")
    bert_retrieved, bert_scores = bert_system.batch_search(
        bert_query_embeddings, k=100, show_progress=True
    )
    
    print("\nLOTT retrieval...")
    lott_retrieved, lott_scores = lott_system.batch_search(
        lott_query_embeddings, k=100, show_progress=True
    )
    
    # Diagnostic: Checking if relevant docs are in index
    print("\n" + "="*80)
    print("DIAGNOSTIC: Checking if relevant docs are in index")
    print("="*80)
    
    total_relevant = 0
    total_in_index = 0
    
    for query_id in query_ids[:10]:
        if query_id in qrels:
            relevant_docs = set(qrels[query_id].keys())
            docs_in_index = sum(1 for doc in relevant_docs if doc in doc_ids)
            total_relevant += len(relevant_docs)
            total_in_index += docs_in_index
            print(f"Query {query_id}: {docs_in_index}/{len(relevant_docs)} relevant docs in index")
    
    if total_relevant > 0:
        coverage = total_in_index / total_relevant * 100
        print(f"\nOverall coverage (first 10 queries): {total_in_index}/{total_relevant} ({coverage:.1f}%)")
    
    # Evaluation
    evaluator = RAGEvaluator(qrels)
    
    bert_metrics = evaluator.evaluate_system('BERT', bert_retrieved, query_ids)
    lott_metrics = evaluator.evaluate_system('LOTT', lott_retrieved, lott_query_ids)
    
    evaluator.print_comparison()
    
    evaluator.plot_comparison()
    
    # Saving results
    evaluator.save_results()
    
    # Benchmarking timing
    timing_results = benchmark_timing(
        bert_system, lott_system,
        bert_query_embeddings, lott_query_embeddings,
        n_queries=min(100, len(queries))
    )
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    
    return {
        'bert_metrics': bert_metrics,
        'lott_metrics': lott_metrics,
        'timing': timing_results
    }


if __name__ == "__main__":
    results = run_full_evaluation()
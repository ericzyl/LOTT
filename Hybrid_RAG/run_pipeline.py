"""
Main pipeline for hybrid RAG system
BERT Retrieval + LOTT Reranking
"""
import argparse
import json
import pickle
from typing import Dict, List
import config
from dataset_loaders import load_dataset
from preprocessing import prepare_bow_data
from lda_module import LDAModule
from bert_retriever import BERTRetriever
from lott_reranker import LOTTReranker
from evaluate_metrics import evaluate_retrieval


def run_experiment(dataset_name: str):
    """Run complete experiment for a dataset"""
    
    config.print_config(dataset_name)
    
    # =========================================================================
    # STEP 1: Load Dataset
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATASET")
    print("="*80)
    
    corpus, queries, qrels = load_dataset(dataset_name)
    
    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: PREPROCESSING")
    print("="*80)
    
    bow_data, vocab, embeddings, doc_ids = prepare_bow_data(corpus, dataset_name)
    
    # Load vocab data for query processing
    cache = config.get_cache_paths(dataset_name)
    with open(cache['vocab'], 'rb') as f:
        vocab_data = pickle.load(f)
    
    # =========================================================================
    # STEP 3: LDA Training
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: LDA TRAINING")
    print("="*80)
    
    lda_module = LDAModule(dataset_name)
    topics, lda_centers, topic_costs = lda_module.train_or_load(bow_data, embeddings, vocab)
    
    # =========================================================================
    # STEP 4: BERT Retrieval
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: BERT RETRIEVAL")
    print("="*80)
    
    bert_retriever = BERTRetriever(dataset_name)
    bert_retriever.build_index(corpus, doc_ids)
    
    query_embeddings, query_ids = bert_retriever.encode_queries(queries)
    
    print(f"\nRetrieving top-{config.K_RETRIEVAL} documents with BERT...")
    bert_retrieved, bert_scores = bert_retriever.retrieve(query_embeddings, k=config.K_RETRIEVAL)
    
    # =========================================================================
    # STEP 5: LOTT Reranking Preparation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: LOTT RERANKING PREPARATION")
    print("="*80)
    
    lott_reranker = LOTTReranker(dataset_name, lda_module)
    lott_reranker.prepare_documents(bow_data)
    lott_reranker.prepare_queries(queries, corpus, vocab_data)
    
    # Create doc_id to index mapping
    doc_id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    # =========================================================================
    # STEP 6: Evaluation
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: EVALUATION")
    print("="*80)
    
    all_results = {}
    
    # Method 1: BERT Only (baseline)
    print("\n### Method 1: BERT Only ###")
    all_results['bert_only'] = evaluate_retrieval(
        bert_retrieved,
        query_ids,
        qrels,
        "BERT Only"
    )
    
    # Method 2: BERT + LOTT Embedding Reranking
    print("\n### Method 2: BERT + LOTT Embedding Reranking ###")
    lott_emb_retrieved = []
    for query_idx in range(len(query_ids)):
        if query_idx % 100 == 0:
            print(f"  Reranking query {query_idx}/{len(query_ids)}...")
        
        # For each K, rerank top-100 to get top-K
        # We'll store full reranked list and evaluate at different K
        reranked = lott_reranker.rerank_embedding_based(
            query_idx,
            bert_retrieved[query_idx],
            doc_id_to_idx,
            k=config.K_RETRIEVAL  # Rerank all 100
        )
        lott_emb_retrieved.append(reranked)
    
    all_results['bert_lott_embedding'] = evaluate_retrieval(
        lott_emb_retrieved,
        query_ids,
        qrels,
        "BERT + LOTT Embedding"
    )
    
    # Method 3: BERT + LOTT OT Reranking
    print("\n### Method 3: BERT + LOTT OT Reranking ###")
    lott_ot_retrieved = []
    for query_idx in range(len(query_ids)):
        if query_idx % 100 == 0:
            print(f"  Reranking query {query_idx}/{len(query_ids)}...")
        
        reranked = lott_reranker.rerank_ot_based(
            query_idx,
            bert_retrieved[query_idx],
            doc_id_to_idx,
            k=config.K_RETRIEVAL  # Rerank all 100
        )
        lott_ot_retrieved.append(reranked)
    
    all_results['bert_lott_ot'] = evaluate_retrieval(
        lott_ot_retrieved,
        query_ids,
        qrels,
        "BERT + LOTT OT"
    )
    
    # =========================================================================
    # STEP 7: Save Results
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: SAVING RESULTS")
    print("="*80)
    
    results_path = config.get_results_path(dataset_name)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for method_name, results in all_results.items():
        print(f"\n{method_name}:")
        for k in config.TOP_K_VALUES:
            print(f"  NDCG@{k}: {results[f'NDCG@{k}']:.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Hybrid RAG: BERT + LOTT Reranking")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=config.AVAILABLE_DATASETS,
        default='trec-covid',
        help='Dataset to use'
    )
    
    args = parser.parse_args()
    
    results = run_experiment(args.dataset)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print(f"\nRun visualize_results.py --dataset {args.dataset} to see plots")


if __name__ == "__main__":
    main()
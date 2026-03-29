"""
run_experiment.py
=================
Compares two retrieval strategies on MS-MARCO:

  Method A – BERT only
    Dense FAISS retrieval over all documents with SBERT, top-K_FINAL.

  Method B – LOTT → BERT hybrid
    1. LOTT FAISS retrieval: fetches top-K_LOTT_RETRIEVAL candidates.
    2. BERT reranker: scores that candidate pool, returns top-K_FINAL.

Both methods are evaluated on the same metrics (P@K, R@K, NDCG@K, MRR, MAP)
and wall-clock timing is recorded for each pipeline stage.
"""

import argparse
import json
import pickle
import time
from typing import Dict, List

import config
from dataset_loader  import load_dataset
from preprocessing   import prepare_bow_data
from lda_module      import LDAModule
from lott_retriever  import LOTTRetriever
from bert_reranker   import BERTReranker
from evaluate_metrics import evaluate_retrieval


# ---------------------------------------------------------------------------

def _timed(label: str, fn, *args, **kwargs):
    """Call fn(*args, **kwargs), print elapsed time, return (result, seconds)."""
    print(f"\n[TIMER] Starting: {label}")
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"[TIMER] {label}: {elapsed:.2f}s")
    return result, elapsed


# ---------------------------------------------------------------------------

def run_experiment(dataset_name: str):
    config.print_config(dataset_name)

    timings: Dict[str, float] = {}

    # =========================================================================
    # STEP 1 – Load Dataset
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATASET")
    print("=" * 80)
    corpus, queries, qrels = load_dataset(dataset_name)
    print(f"Corpus: {len(corpus)} docs | Queries: {len(queries)} | Qrels: {len(qrels)}")

    # =========================================================================
    # STEP 2 – Preprocessing
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING")
    print("=" * 80)
    bow_data, vocab, embeddings, doc_ids = prepare_bow_data(corpus, dataset_name)

    cache = config.get_cache_paths(dataset_name)
    with open(cache['vocab'], 'rb') as f:
        vocab_data = pickle.load(f)

    # =========================================================================
    # STEP 3 – LDA Training
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: LDA TRAINING")
    print("=" * 80)
    lda_module = LDAModule(dataset_name)
    _, lda_train_time = _timed(
        "LDA train_or_load",
        lda_module.train_or_load, bow_data, embeddings, vocab
    )
    timings['lda_train'] = lda_train_time

    # =========================================================================
    # STEP 4 – Prepare shared BERT components (doc embeddings)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: BERT DOCUMENT ENCODING")
    print("=" * 80)
    bert = BERTReranker(dataset_name)
    _, bert_doc_enc_time = _timed(
        "BERT doc encoding",
        bert.prepare_documents, corpus, doc_ids
    )
    timings['bert_doc_encoding'] = bert_doc_enc_time

    # Encode queries once, shared by both methods
    print("\n" + "=" * 80)
    print("STEP 5: BERT QUERY ENCODING")
    print("=" * 80)
    (bert_query_embs, query_ids), bert_q_enc_time = _timed(
        "BERT query encoding",
        bert.encode_queries, queries
    )
    timings['bert_query_encoding'] = bert_q_enc_time

    # =========================================================================
    # STEP 6 – Build LOTT index and encode queries
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: LOTT INDEX BUILDING")
    print("=" * 80)
    lott = LOTTRetriever(dataset_name, lda_module)
    _, lott_index_time = _timed(
        "LOTT index build",
        lott.build_index, bow_data, doc_ids
    )
    timings['lott_index_build'] = lott_index_time

    print("\n" + "=" * 80)
    print("STEP 7: LOTT QUERY ENCODING")
    print("=" * 80)
    (lott_query_embs, lott_query_ids), lott_q_enc_time = _timed(
        "LOTT query encoding",
        lott.encode_queries, queries, vocab_data
    )
    timings['lott_query_encoding'] = lott_q_enc_time

    # Sanity-check: both pipelines share the same query order
    assert query_ids == lott_query_ids, "Query ID mismatch between BERT and LOTT!"

    # =========================================================================
    # STEP 8 – METHOD A: BERT-only retrieval
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: METHOD A – BERT-ONLY RETRIEVAL")
    print("=" * 80)
    bert.build_full_index()
    (bert_retrieved, bert_scores), bert_retrieval_time = _timed(
        "BERT-only retrieval",
        bert.retrieve_full, bert_query_embs, config.K_FINAL
    )
    timings['bert_retrieval'] = bert_retrieval_time

    # =========================================================================
    # STEP 9 – METHOD B: LOTT retrieval → BERT reranking
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 9: METHOD B – LOTT RETRIEVAL")
    print("=" * 80)
    (lott_candidates, _), lott_retrieval_time = _timed(
        "LOTT first-stage retrieval",
        lott.retrieve, lott_query_embs, config.K_LOTT_RETRIEVAL
    )
    timings['lott_retrieval'] = lott_retrieval_time

    print("\n" + "=" * 80)
    print("STEP 10: METHOD B – BERT RERANKING OF LOTT CANDIDATES")
    print("=" * 80)
    (hybrid_retrieved, hybrid_scores), bert_rerank_time = _timed(
        "BERT reranking",
        bert.rerank, bert_query_embs, lott_candidates, config.K_FINAL
    )
    timings['bert_reranking'] = bert_rerank_time

    timings['hybrid_total'] = lott_retrieval_time + bert_rerank_time

    # =========================================================================
    # STEP 11 – Evaluation
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 11: EVALUATION")
    print("=" * 80)

    bert_metrics   = evaluate_retrieval(bert_retrieved,   query_ids, qrels, "BERT only")
    hybrid_metrics = evaluate_retrieval(hybrid_retrieved, query_ids, qrels, "LOTT + BERT")

    # =========================================================================
    # STEP 12 – Save results
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 12: SAVING RESULTS")
    print("=" * 80)

    all_results = {
        'bert_only':    bert_metrics,
        'lott_bert':    hybrid_metrics,
        'timings':      timings,
        'config': {
            'K_LOTT_RETRIEVAL': config.K_LOTT_RETRIEVAL,
            'K_FINAL':          config.K_FINAL,
            'N_TOPICS':         config.N_TOPICS,
            'BERT_MODEL':       config.BERT_MODEL,
        },
    }

    results_path = config.get_results_path(dataset_name)
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved → {results_path}")

    # =========================================================================
    # Summary table
    # =========================================================================
    _print_summary(bert_metrics, hybrid_metrics, timings)

    return all_results


# ---------------------------------------------------------------------------

def _print_summary(bert_m: dict, hybrid_m: dict, timings: dict):
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Metric':<12} {'BERT only':>12} {'LOTT+BERT':>12} {'Δ':>10}")
    print("-" * 50)
    keys = [f'NDCG@{k}' for k in config.TOP_K_VALUES] + ['MRR', 'MAP']
    for key in keys:
        b = bert_m.get(key, 0)
        h = hybrid_m.get(key, 0)
        delta = h - b
        sym   = '+' if delta >= 0 else ''
        print(f"{key:<12} {b:>12.4f} {h:>12.4f} {sym}{delta:>9.4f}")

    print("\n  TIMING")
    print(f"  BERT-only retrieval  : {timings.get('bert_retrieval', 0):.2f}s")
    print(f"  LOTT retrieval       : {timings.get('lott_retrieval', 0):.2f}s")
    print(f"  BERT reranking       : {timings.get('bert_reranking', 0):.2f}s")
    print(f"  Hybrid total         : {timings.get('hybrid_total', 0):.2f}s")
    print("=" * 80)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid RAG: LOTT first-stage retrieval + BERT reranking vs BERT-only"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=config.AVAILABLE_DATASETS,
        default='msmarco',
        help='Dataset to use (default: msmarco)',
    )
    args = parser.parse_args()
    run_experiment(args.dataset)
    print(f"\nRun  python visualize_results.py --dataset {args.dataset}  to see plots.")


if __name__ == "__main__":
    main()
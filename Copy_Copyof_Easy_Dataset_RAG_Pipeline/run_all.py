import sys
import time
from pathlib import Path

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def check_glove_embeddings():
    import config
    
    if not Path(config.GLOVE_PATH).exists():
        print("ERROR: GloVe embeddings not found!")
        print(f"Expected location: {config.GLOVE_PATH}")
        print("\nPlease download GloVe embeddings:")
        print("1. Go to: https://nlp.stanford.edu/projects/glove/")
        print("2. Download: glove.6B.zip")
        print("3. Extract glove.6B.300d.txt to data/glove.6B/")
        return False
    return True

def run_pipeline():
    print_section("STEP 1/5: Running Complete Pipeline")
    
    from pipeline import run_full_pipeline
    
    try:
        pipeline_data = run_full_pipeline()
        print("Pipeline completed successfully!")
        return True
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_bert_rag():
    print_section("STEP 2/5: Building BERT RAG System")
    
    from rag_bert import build_bert_rag_system
    
    try:
        bert_system = build_bert_rag_system()
        print("BERT RAG system built successfully!")
        return True
    except Exception as e:
        print(f"BERT RAG build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def build_lott_rag():
    print_section("STEP 3/5: Building LOTT RAG System")
    
    from rag_lott import build_lott_rag_system
    
    try:
        lott_system = build_lott_rag_system()
        print("LOTT RAG system built successfully!")
        return True
    except Exception as e:
        print(f"LOTT RAG build failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_evaluation():
    print_section("STEP 4/5: Running Evaluation")
    
    from evaluate_rag import run_full_evaluation
    
    try:
        results = run_full_evaluation()
        print("Evaluation completed successfully!")
        return True, results
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def print_summary(results):
    print_section("STEP 5/5: Summary")
    
    if results is None:
        print("Evaluation did not complete successfully.")
        return
    
    print("RETRIEVAL QUALITY COMPARISON:")
    print("-" * 40)
    
    bert_metrics = results['bert_metrics']
    lott_metrics = results['lott_metrics']
    
    print(f"{'Metric':<15} {'BERT':<12} {'LOTT':<12}")
    print("-" * 40)
    
    for metric in ['P@1', 'P@5', 'P@10', 'MAP', 'MRR']:
        if metric in bert_metrics and metric in lott_metrics:
            print(f"{metric:<15} {bert_metrics[metric]:<12.4f} {lott_metrics[metric]:<12.4f}")
    
    print("\nTIMING COMPARISON:")
    print("-" * 40)
    
    timing = results['timing']
    print(f"{'System':<15} {'Per Query (ms)':<20} {'Throughput (q/s)':<20}")
    print("-" * 40)
    print(f"{'BERT':<15} {timing['BERT']['per_query']*1000:<20.2f} "
          f"{timing['BERT']['queries_per_second']:<20.1f}")
    print(f"{'LOTT':<15} {timing['LOTT']['per_query']*1000:<20.2f} "
          f"{timing['LOTT']['queries_per_second']:<20.1f}")
    
    print("\nRESULTS SAVED TO:")
    print(f"  - Metrics: results/evaluation_metrics.json")
    print(f"  - Timing: results/timing_results.json")
    print(f"  - Plots: results/comparison_plot.png")

def main():
    start_time = time.time()
    
    print("\n" + "="*80)
    print("  RAG SYSTEM COMPARISON: BERT vs LOTT")
    print("  Complete Pipeline Execution")
    print("="*80)
    
    # Check prerequisites
    print("\nChecking prerequisites...")
    if not check_glove_embeddings():
        sys.exit(1)
    
    print("✓ All prerequisites satisfied!\n")
    
    # Run pipeline steps
    steps = [
        ("Pipeline", run_pipeline),
        ("BERT RAG", build_bert_rag),
        ("LOTT RAG", build_lott_rag),
        ("Evaluation", run_evaluation),
    ]
    
    results = None
    
    for step_name, step_func in steps:
        success = step_func()
        
        if step_name == "Evaluation" and success:
            success, results = success
        
        if not success:
            print(f"\n✗ Failed at step: {step_name}")
            print("Please fix the error and try again.")
            sys.exit(1)
    
    # Print summary
    print_summary(results)
    
    # Total time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")
    
    print("\n" + "="*80)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
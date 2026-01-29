import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import config


def plot_comparison(dataset_name: str):

    # Load results
    results_path = config.get_results_path(dataset_name)
    with open(results_path, 'r') as f:
        all_results = json.load(f)
    
    methods = ['bert_only', 'bert_lott_embedding', 'bert_lott_ot']
    method_labels = ['BERT Only', 'BERT + LOTT Emb', 'BERT + LOTT OT']
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, Red, Green
    
    # Create figure with subplots for each metric
    metrics = ['P', 'R', 'NDCG']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Retrieval Performance Comparison - {dataset_name.upper()}', fontsize=16, fontweight='bold')
    
    x = np.arange(len(config.TOP_K_VALUES))
    width = 0.25
    
    for metric_idx, metric_name in enumerate(metrics):
        ax = axes[metric_idx]
        
        for method_idx, method in enumerate(methods):
            values = [all_results[method][f'{metric_name}@{k}'] for k in config.TOP_K_VALUES]
            offset = width * (method_idx - 1)
            ax.bar(x + offset, values, width, label=method_labels[method_idx], 
                   color=colors[method_idx], alpha=0.8)
        
        ax.set_xlabel('K', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric_name} Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_name}@K', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(config.TOP_K_VALUES)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.1))  # Ensure y-axis starts at 0
    
    plt.tight_layout()
    
    # Save plot
    plot_path = config.get_plot_path(dataset_name)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")
    plt.close()
    
    # Create a second plot for MRR comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    mrr_values = [all_results[method]['MRR'] for method in methods]
    bars = ax.bar(range(len(methods)), mrr_values, color=colors, alpha=0.8, width=0.6)
    
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('MRR Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean Reciprocal Rank (MRR) Comparison - {dataset_name.upper()}', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(mrr_values) * 1.2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    mrr_plot_path = config.RESULTS_DIR / f"{dataset_name}_mrr_comparison.png"
    plt.savefig(mrr_plot_path, dpi=300, bbox_inches='tight')
    print(f"MRR plot saved to {mrr_plot_path}")
    plt.close()
    
    # Print summary table
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY - {dataset_name.upper()}")
    print("="*80)
    print(f"{'Method':<25} {'NDCG@5':<12} {'NDCG@10':<12} {'P@5':<12} {'MRR':<12}")
    print("-"*80)
    
    for method, label in zip(methods, method_labels):
        results = all_results[method]
        print(f"{label:<25} {results['NDCG@5']:<12.4f} {results['NDCG@10']:<12.4f} "
              f"{results['P@5']:<12.4f} {results['MRR']:<12.4f}")
    
    print("="*80)
    
    # Calculate improvements
    print("\nIMPROVEMENT OVER BERT ONLY:")
    print("-"*80)
    
    for method, label in zip(methods[1:], method_labels[1:]):
        print(f"\n{label}:")
        for metric in ['NDCG@5', 'NDCG@10', 'P@5', 'MRR']:
            bert_val = all_results['bert_only'][metric]
            method_val = all_results[method][metric]
            if bert_val > 0:
                improvement = ((method_val - bert_val) / bert_val) * 100
                symbol = "+" if improvement > 0 else ""
                print(f"  {metric}: {symbol}{improvement:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Visualize hybrid RAG results")
    parser.add_argument(
        '--dataset',
        type=str,
        choices=config.AVAILABLE_DATASETS,
        default='trec-covid',
        help='Dataset to visualize'
    )
    
    args = parser.parse_args()
    
    plot_comparison(args.dataset)
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
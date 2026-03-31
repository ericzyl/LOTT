"""
visualize_results.py
====================
Produces four plots after run_experiment.py has finished:

  1. Precision@K, Recall@K, NDCG@K grouped bar charts
  2. MRR + MAP side-by-side bars
  3. Timing breakdown stacked bar chart
  4. Per-query NDCG@10 scatter / improvement plot  (optional)

All plots are saved to the results/ directory.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

import config

# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

COLORS = {
    'bert':   '#3B82F6',   # blue
    'hybrid': '#10B981',   # green
}
HATCHES = {'bert': '', 'hybrid': '//'}

plt.rcParams.update({
    'figure.dpi':       150,
    'axes.spines.top':  False,
    'axes.spines.right': False,
    'axes.grid':        True,
    'axes.grid.axis':   'y',
    'grid.alpha':       0.3,
    'font.size':        11,
})


def _bar_pair(ax, x, v_bert, v_hybrid, width=0.32, label_bert='BERT only', label_hybrid='LOTT + BERT'):
    ax.bar(x - width / 2, v_bert,   width, color=COLORS['bert'],
           hatch=HATCHES['bert'],   label=label_bert,   alpha=0.85, edgecolor='white')
    ax.bar(x + width / 2, v_hybrid, width, color=COLORS['hybrid'],
           hatch=HATCHES['hybrid'], label=label_hybrid, alpha=0.85, edgecolor='white')


def _annotate_bars(ax, bars_vals, x_positions, offset_sign=1, width=0.32):
    for xpos, val in zip(x_positions, bars_vals):
        ax.text(xpos, val + 0.002, f'{val:.3f}',
                ha='center', va='bottom', fontsize=8, color='#555')


# ---------------------------------------------------------------------------
# Plot 1 – P@K / R@K / NDCG@K
# ---------------------------------------------------------------------------

def plot_prk(results: dict, dataset_name: str):
    ks     = config.TOP_K_VALUES
    x      = np.arange(len(ks))
    bert   = results['bert_only']
    hybrid = results['lott_bert']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Retrieval metrics – {dataset_name.upper()}  '
                 f'(BERT only vs LOTT→BERT)', fontsize=13, fontweight='bold')

    for ax, prefix, title in zip(
        axes,
        ['P', 'R', 'NDCG'],
        ['Precision@K', 'Recall@K', 'NDCG@K'],
    ):
        vb = [bert.get(f'{prefix}@{k}', 0)   for k in ks]
        vh = [hybrid.get(f'{prefix}@{k}', 0) for k in ks]
        _bar_pair(ax, x, vb, vh)
        ax.set_xticks(x); ax.set_xticklabels(ks)
        ax.set_xlabel('K'); ax.set_ylabel('Score'); ax.set_title(title)
        ax.legend(fontsize=9); ax.set_ylim(0, max(max(vb), max(vh)) * 1.25 + 0.01)

    plt.tight_layout()
    p = config.get_plot_path(dataset_name)
    plt.savefig(p, bbox_inches='tight')
    print(f"Saved metric plot → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 2 – MRR and MAP
# ---------------------------------------------------------------------------

def plot_mrr_map(results: dict, dataset_name: str):
    bert   = results['bert_only']
    hybrid = results['lott_bert']

    metrics = ['MRR', 'MAP']
    labels  = ['MRR', 'MAP']
    vb      = [bert.get(m, 0)   for m in metrics]
    vh      = [hybrid.get(m, 0) for m in metrics]
    x       = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f'MRR & MAP – {dataset_name.upper()}', fontsize=13, fontweight='bold')
    _bar_pair(ax, x, vb, vh)

    for i, (b, h) in enumerate(zip(vb, vh)):
        ax.text(i - 0.16, b + 0.003, f'{b:.4f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + 0.16, h + 0.003, f'{h:.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_ylim(0, max(max(vb), max(vh)) * 1.3 + 0.01)
    ax.legend(fontsize=10)
    plt.tight_layout()

    # p = config.RESULTS_DIR / f"{dataset_name}_mrr_map.png"
    p = config.get_mrr_map_plot_path(dataset_name)
    plt.savefig(p, bbox_inches='tight')
    print(f"Saved MRR/MAP plot → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 3 – Timing breakdown
# ---------------------------------------------------------------------------

def plot_timing(results: dict, dataset_name: str):
    timings = results.get('timings', {})

    # BERT pipeline: just retrieval time
    bert_ret    = timings.get('bert_retrieval', 0)

    # Hybrid pipeline: LOTT retrieval + BERT reranking
    lott_ret    = timings.get('lott_retrieval', 0)
    bert_rerank = timings.get('bert_reranking', 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Query-time timing breakdown – {dataset_name.upper()}',
                 fontsize=13, fontweight='bold')

    # ---- Left: stacked bar ----
    ax = axes[0]
    methods = ['BERT only', 'LOTT + BERT']
    bot = np.zeros(2)

    segments = [
        ('BERT retrieval',  [bert_ret,    0          ], COLORS['bert']),
        ('LOTT retrieval',  [0,           lott_ret   ], '#F59E0B'),
        ('BERT reranking',  [0,           bert_rerank], COLORS['hybrid']),
    ]

    for label, vals, color in segments:
        ax.bar(methods, vals, bottom=bot, label=label, color=color, alpha=0.85, edgecolor='white')
        bot += np.array(vals)

    for i, total in enumerate(bot):
        ax.text(i, total + 0.05, f'{total:.2f}s', ha='center', va='bottom', fontsize=10)

    ax.set_ylabel('Wall-clock seconds')
    ax.set_title('Total query-time (seconds)')
    ax.legend(fontsize=9)

    # ---- Right: per-query latency (ms) ----
    ax2 = axes[1]
    n_queries = results['bert_only'].get('num_queries', 1)
    bert_ms   = bert_ret    / max(n_queries, 1) * 1000
    hybrid_ms = (lott_ret + bert_rerank) / max(n_queries, 1) * 1000

    bars = ax2.bar(
        ['BERT only', 'LOTT + BERT'],
        [bert_ms, hybrid_ms],
        color=[COLORS['bert'], COLORS['hybrid']],
        alpha=0.85, edgecolor='white', width=0.45,
    )
    for bar, val in zip(bars, [bert_ms, hybrid_ms]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f'{val:.2f} ms', ha='center', va='bottom', fontsize=10)

    ax2.set_ylabel('ms per query')
    ax2.set_title('Avg per-query latency (ms)')
    ax2.set_ylim(0, max(bert_ms, hybrid_ms) * 1.3 + 0.5)

    plt.tight_layout()
    p = config.get_timing_plot_path(dataset_name)
    plt.savefig(p, bbox_inches='tight')
    print(f"Saved timing plot → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 4 – Improvement bar chart (Δ over BERT-only)
# ---------------------------------------------------------------------------

def plot_delta(results: dict, dataset_name: str):
    bert   = results['bert_only']
    hybrid = results['lott_bert']

    keys   = [f'NDCG@{k}' for k in config.TOP_K_VALUES] + ['MRR', 'MAP']
    deltas = [hybrid.get(k, 0) - bert.get(k, 0) for k in keys]
    colors = ['#10B981' if d >= 0 else '#EF4444' for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(keys, deltas, color=colors, alpha=0.85, edgecolor='white')
    ax.axhline(0, color='#6B7280', linewidth=0.8)
    ax.set_ylabel('LOTT+BERT  −  BERT only')
    ax.set_title(f'Absolute improvement of LOTT+BERT over BERT only – {dataset_name.upper()}',
                 fontsize=12, fontweight='bold')

    for i, (k, d) in enumerate(zip(keys, deltas)):
        sign = '+' if d >= 0 else ''
        ax.text(i, d + (0.001 if d >= 0 else -0.001),
                f'{sign}{d:.4f}', ha='center',
                va='bottom' if d >= 0 else 'top', fontsize=9)

    plt.tight_layout()
    # p = config.RESULTS_DIR / f"{dataset_name}_delta.png"
    p = config.get_delta_plot_path(dataset_name)
    plt.savefig(p, bbox_inches='tight')
    print(f"Saved delta plot → {p}")
    plt.close()


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------

def print_table(results: dict, dataset_name: str):
    bert   = results['bert_only']
    hybrid = results['lott_bert']
    timings = results.get('timings', {})

    print("\n" + "=" * 80)
    print(f"RESULTS SUMMARY – {dataset_name.upper()}")
    print("=" * 80)
    keys = [f'P@{k}' for k in config.TOP_K_VALUES] + \
           [f'R@{k}' for k in config.TOP_K_VALUES] + \
           [f'NDCG@{k}' for k in config.TOP_K_VALUES] + \
           ['MRR', 'MAP']
    print(f"{'Metric':<12} {'BERT only':>12} {'LOTT+BERT':>12} {'Δ':>10} {'Δ %':>8}")
    print("-" * 60)
    for key in keys:
        b = bert.get(key, 0)
        h = hybrid.get(key, 0)
        d = h - b
        pct = (d / b * 100) if b > 0 else 0
        sym = '+' if d >= 0 else ''
        print(f"{key:<12} {b:>12.4f} {h:>12.4f} {sym}{d:>9.4f} {sym}{pct:>6.1f}%")

    print("\n  TIMING")
    print(f"  BERT-only retrieval  : {timings.get('bert_retrieval', 0):.3f}s")
    print(f"  LOTT retrieval       : {timings.get('lott_retrieval', 0):.3f}s")
    print(f"  BERT reranking       : {timings.get('bert_reranking', 0):.3f}s")
    print(f"  Hybrid total         : {timings.get('hybrid_total', 0):.3f}s")
    print("=" * 80)


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise LOTT+BERT experiment results")
    parser.add_argument(
        '--dataset', type=str, choices=config.AVAILABLE_DATASETS,
        default='msmarco', help='Dataset to visualise',
    )
    args = parser.parse_args()

    results_path = config.get_results_path(args.dataset)
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Run  python run_experiment.py  first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    print_table(results, args.dataset)
    plot_prk(results, args.dataset)
    plot_mrr_map(results, args.dataset)
    plot_timing(results, args.dataset)
    plot_delta(results, args.dataset)
    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
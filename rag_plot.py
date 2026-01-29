import matplotlib.pyplot as plt
import numpy as np
import json

data = {
  "BERT": {
    "P@1": 0.74, "R@1": 0.0006345579056078256, "NDCG@1": 0.57,
    "P@5": 0.632, "R@5": 0.0027310151296788886, "NDCG@5": 0.48884875382586507,
    "P@10": 0.56, "R@10": 0.004869512777414141, "NDCG@10": 0.44996814193679696,
    "P@20": 0.524, "R@20": 0.009124449061942947, "NDCG@20": 0.4227901632948637,
    "MRR": 0.8148333333333333, "MAP": 0.5363900884911105
  },
  "LOTT": {
    "P@1": 0.14, "R@1": 0.00013574333083072134, "NDCG@1": 0.05,
    "P@5": 0.14, "R@5": 0.0005962990223764033, "NDCG@5": 0.06715294821989794,
    "P@10": 0.108, "R@10": 0.0009077332305482217, "NDCG@10": 0.06042561885826678,
    "P@20": 0.088, "R@20": 0.0014686388660216088, "NDCG@20": 0.051425528597109305,
    "MRR": 0.26632209498971376, "MAP": 0.1822325890979785
  }
}

colors = ['#1f77b4', '#aec7e8', '#9467bd', '#1B9B52', '#E6873E',
          '#EA1A47', '#ff9896', '#e377c2', '#bcbd22', '#17becf']

methods = list(data.keys())
method_colors = {'BERT': colors[5], 'LOTT': colors[9]}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RAG Evaluation Metrics on TREC-COVID Dataset', fontsize=18, fontweight='bold', y=0.995)

# Precision@K comparison
ax1 = axes[0, 0]
k_values = [1, 5, 10, 20]
x = np.arange(len(k_values))
width = 0.35

for i, method in enumerate(methods):
    p_values = [data[method][f'P@{k}'] for k in k_values]
    ax1.bar(x + i*width, p_values, width, label=method, color=method_colors[method], alpha=0.8)

ax1.set_xlabel('K', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision@K', fontsize=12, fontweight='bold')
ax1.set_title('Precision at Different K Values', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width/2)
ax1.set_xticklabels([f'{k}' for k in k_values])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 1])

# NDCG@K comparison
ax2 = axes[0, 1]
for i, method in enumerate(methods):
    ndcg_values = [data[method][f'NDCG@{k}'] for k in k_values]
    ax2.bar(x + i*width, ndcg_values, width, label=method, color=method_colors[method], alpha=0.8)

ax2.set_xlabel('K', fontsize=12, fontweight='bold')
ax2.set_ylabel('NDCG@K', fontsize=12, fontweight='bold')
ax2.set_title('NDCG at Different K Values', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width/2)
ax2.set_xticklabels([f'{k}' for k in k_values])
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 1])

# Recall@K comparison (scaled for visibility)
ax3 = axes[1, 0]
for i, method in enumerate(methods):
    r_values = [data[method][f'R@{k}'] * 100 for k in k_values]  # Scale to percentage
    ax3.bar(x + i*width, r_values, width, label=method, color=method_colors[method], alpha=0.8)

ax3.set_xlabel('K', fontsize=12, fontweight='bold')
ax3.set_ylabel('Recall@K (%)', fontsize=12, fontweight='bold')
ax3.set_title('Recall at Different K Values (scaled)', fontsize=13, fontweight='bold')
ax3.set_xticks(x + width/2)
ax3.set_xticklabels([f'{k}' for k in k_values])
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# Overall Metrics (MRR and MAP)
ax4 = axes[1, 1]
overall_metrics = ['MRR', 'MAP']
x_overall = np.arange(len(overall_metrics))

for i, method in enumerate(methods):
    values = [data[method][metric] for metric in overall_metrics]
    ax4.bar(x_overall + i*width, values, width, label=method, color=method_colors[method], alpha=0.8)

ax4.set_ylabel('Score', fontsize=12, fontweight='bold')
ax4.set_title('Overall Performance Metrics', fontsize=13, fontweight='bold')
ax4.set_xticks(x_overall + width/2)
ax4.set_xticklabels(overall_metrics, fontsize=11)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('rag_evaluation_trec_covid.png', dpi=300, bbox_inches='tight')
print("Saved plot to: rag_evaluation_trec_covid.png")
# plt.show()

# Combined line plot showing trends
fig2, ax = plt.subplots(1, 1, figsize=(10, 6))

k_values = [1, 5, 10, 20]
for method in methods:
    p_values = [data[method][f'P@{k}'] for k in k_values]
    ndcg_values = [data[method][f'NDCG@{k}'] for k in k_values]
    
    ax.plot(k_values, p_values, marker='o', linewidth=2.5, markersize=8, 
            label=f'{method} - Precision', color=method_colors[method], linestyle='-')
    ax.plot(k_values, ndcg_values, marker='s', linewidth=2.5, markersize=8, 
            label=f'{method} - NDCG', color=method_colors[method], linestyle='--', alpha=0.7)

ax.set_xlabel('K', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Precision and NDCG Trends Across K Values\nTREC-COVID Dataset', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 1])
ax.set_xticks(k_values)

plt.tight_layout()
plt.savefig('rag_evaluation_trends_trec_covid.png', dpi=300, bbox_inches='tight')
print("Saved plot to: rag_evaluation_trends_trec_covid.png")
# plt.show()
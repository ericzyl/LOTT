import matplotlib.pyplot as plt
import numpy as np

# Data from Hybrid RAG experiments
data = {
  "BERT Only": {
    "P@1": 0.74, "R@1": 0.0006, "NDCG@1": 0.61,
    "P@5": 0.632, "R@5": 0.0027, "NDCG@5": 0.5223,
    "P@10": 0.598, "R@10": 0.0052, "NDCG@10": 0.4771,
    "P@20": 0.549, "R@20": 0.0095, "NDCG@20": 0.4481,
    "MRR": 0.8185
  },
  "BERT + LOTT Embedding": {
    "P@1": 0.34, "R@1": 0.0003, "NDCG@1": 0.24,
    "P@5": 0.404, "R@5": 0.0018, "NDCG@5": 0.251,
    "P@10": 0.37, "R@10": 0.0033, "NDCG@10": 0.2493,
    "P@20": 0.355, "R@20": 0.0063, "NDCG@20": 0.2488,
    "MRR": 0.542
  },
  "BERT + LOTT OT": {
    "P@1": 0.40, "R@1": 0.0004, "NDCG@1": 0.22,
    "P@5": 0.476, "R@5": 0.0020, "NDCG@5": 0.2785,
    "P@10": 0.47, "R@10": 0.0041, "NDCG@10": 0.3061,
    "P@20": 0.47, "R@20": 0.0082, "NDCG@20": 0.3233,
    "MRR": 0.581
  }
}

colors = ['#1f77b4', '#1B9B52', '#E6873E']
methods = list(data.keys())
method_colors = {methods[i]: colors[i] for i in range(len(methods))}

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle('Hybrid RAG Performance on TREC-COVID Dataset', fontsize=18, fontweight='bold', y=0.995)

k_values = [1, 5, 10, 20]
x = np.arange(len(k_values))
width = 0.25

# Precision@K comparison
ax1 = axes[0, 0]
for i, method in enumerate(methods):
    p_values = [data[method][f'P@{k}'] for k in k_values]
    ax1.bar(x + i*width, p_values, width, label=method, color=method_colors[method], alpha=0.85)

ax1.set_xlabel('K', fontsize=12, fontweight='bold')
ax1.set_ylabel('Precision@K', fontsize=12, fontweight='bold')
ax1.set_title('Precision at Different K Values', fontsize=13, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels([f'{k}' for k in k_values])
ax1.legend(fontsize=10, loc='upper right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0, 0.85])

# NDCG@K comparison
ax2 = axes[0, 1]
for i, method in enumerate(methods):
    ndcg_values = [data[method][f'NDCG@{k}'] for k in k_values]
    ax2.bar(x + i*width, ndcg_values, width, label=method, color=method_colors[method], alpha=0.85)

ax2.set_xlabel('K', fontsize=12, fontweight='bold')
ax2.set_ylabel('NDCG@K', fontsize=12, fontweight='bold')
ax2.set_title('NDCG at Different K Values', fontsize=13, fontweight='bold')
ax2.set_xticks(x + width)
ax2.set_xticklabels([f'{k}' for k in k_values])
ax2.legend(fontsize=10, loc='upper right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0, 0.7])

# Recall@K comparison (scaled to percentage)
ax3 = axes[1, 0]
for i, method in enumerate(methods):
    r_values = [data[method][f'R@{k}'] * 100 for k in k_values]
    ax3.bar(x + i*width, r_values, width, label=method, color=method_colors[method], alpha=0.85)

ax3.set_xlabel('K', fontsize=12, fontweight='bold')
ax3.set_ylabel('Recall@K (%)', fontsize=12, fontweight='bold')
ax3.set_title('Recall at Different K Values', fontsize=13, fontweight='bold')
ax3.set_xticks(x + width)
ax3.set_xticklabels([f'{k}' for k in k_values])
ax3.legend(fontsize=10, loc='upper right')
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 4. MRR comparison
ax4 = axes[1, 1]
mrr_values = [data[method]['MRR'] for method in methods]
bars = ax4.bar(methods, mrr_values, color=[method_colors[m] for m in methods], alpha=0.85, width=0.6)

for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylabel('MRR', fontsize=12, fontweight='bold')
ax4.set_title('Mean Reciprocal Rank (MRR)', fontsize=13, fontweight='bold')
ax4.set_xticklabels(methods, rotation=15, ha='right', fontsize=10)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0, 1])

plt.tight_layout()
plt.savefig('hybrid_rag_trec_covid_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: hybrid_rag_trec_covid_comparison.png")

# Line Plot showing Trends
fig2, ax = plt.subplots(1, 1, figsize=(12, 7))

for method in methods:
    p_values = [data[method][f'P@{k}'] for k in k_values]
    ndcg_values = [data[method][f'NDCG@{k}'] for k in k_values]
    
    ax.plot(k_values, p_values, marker='o', linewidth=2.5, markersize=9, 
            label=f'{method} - Precision', color=method_colors[method], linestyle='-')
    ax.plot(k_values, ndcg_values, marker='s', linewidth=2.5, markersize=9, 
            label=f'{method} - NDCG', color=method_colors[method], linestyle='--', alpha=0.7)

ax.set_xlabel('K', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Precision and NDCG Trends: Hybrid RAG on TREC-COVID', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='best', ncol=2)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 0.8])
ax.set_xticks(k_values)

plt.tight_layout()
plt.savefig('hybrid_rag_trends_trec_covid.png', dpi=300, bbox_inches='tight')
print("Saved: hybrid_rag_trends_trec_covid.png")

# Focused NDCG comparison
fig3, ax = plt.subplots(1, 1, figsize=(10, 6))

for method in methods:
    ndcg_values = [data[method][f'NDCG@{k}'] for k in k_values]
    ax.plot(k_values, ndcg_values, marker='o', linewidth=3, markersize=10, 
            label=method, color=method_colors[method])

ax.set_xlabel('K', fontsize=14, fontweight='bold')
ax.set_ylabel('NDCG@K', fontsize=14, fontweight='bold')
ax.set_title('NDCG Comparison: Impact of LOTT Reranking\nTREC-COVID Dataset', 
             fontsize=15, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 0.7])
ax.set_xticks(k_values)

ax.annotate('BERT Only\nperforms best', 
            xy=(1, data["BERT Only"]["NDCG@1"]), xytext=(2, 0.55),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
            fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('hybrid_rag_ndcg_focus_trec_covid.png', dpi=300, bbox_inches='tight')
print("Saved: hybrid_rag_ndcg_focus_trec_covid.png")

# plt.show()
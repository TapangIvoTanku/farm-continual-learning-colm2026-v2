"""
FARM Paper Figure Generation
Generates all publication-quality figures for the ICLR 2026 paper.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

RESULTS_DIR = "/home/ubuntu/FARM_paper/results"
OUT_DIR = "/home/ubuntu/FARM_paper/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all results ─────────────────────────────────────────────────────────
METHODS = {
    'full_finetune': 'Full Fine-Tune',
    'ewc': 'EWC',
    'o_lora': 'O-LoRA',
    'lora_replay': 'LoRA+Replay',
    'magmax': 'MagMax',
    'codyre': 'CoDyRA',
    'farm': 'FARM (Ours)',
}

# Short names for x-axis labels to prevent overlap
METHODS_SHORT = {
    'full_finetune': 'FT',
    'ewc': 'EWC',
    'o_lora': 'O-LoRA',
    'lora_replay': 'Replay',
    'magmax': 'MagMax',
    'codyre': 'CoDyRA',
    'farm': 'FARM',
}

COLORS = {
    'full_finetune': '#e74c3c',
    'ewc': '#3498db',
    'o_lora': '#2ecc71',
    'lora_replay': '#9b59b6',
    'magmax': '#e67e22',
    'codyre': '#1abc9c',
    'farm': '#c0392b',
}

MARKERS = {
    'full_finetune': 'o',
    'ewc': 's',
    'o_lora': '^',
    'lora_replay': 'D',
    'magmax': 'v',
    'codyre': 'P',
    'farm': '*',
}

TASKS = ['T1\nXSum', 'T2\nCNN/DM', 'T3\nMedQA', 'T4\nGSM8K', 'T5\nHumanEval']
TASK_KEYS = ['T1_xsum', 'T2_cnn_dailymail', 'T3_medqa', 'T4_gsm8k', 'T5_humaneval']

results = {}
for method_key in METHODS:
    path = os.path.join(RESULTS_DIR, f"{method_key}_results.json")
    if os.path.exists(path):
        with open(path) as f:
            results[method_key] = json.load(f)

# ── Figure 1: Main Results Bar Chart ─────────────────────────────────────────
def fig1_main_results():
    # Use a taller, wider figure to give x-axis labels room
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    metrics = [('acc', 'ACC ↑', 'Average Accuracy'),
               ('bwt', 'BWT ↑', 'Backward Transfer\n(less forgetting = higher)'),
               ('fwt', 'FWT ↑', 'Forward Transfer')]

    method_order = sorted(results.keys(),
                          key=lambda m: results[m]['final_cl_metrics'].get('acc', 0),
                          reverse=True)

    for ax, (metric, ylabel, title) in zip(axes, metrics):
        vals = [results[m]['final_cl_metrics'].get(metric, 0) for m in method_order]
        colors = [COLORS[m] for m in method_order]
        # Use short labels to prevent overlap
        labels = [METHODS_SHORT[m] for m in method_order]

        bars = ax.bar(range(len(method_order)), vals, color=colors,
                      edgecolor='white', linewidth=0.8, zorder=3,
                      width=0.55)

        # Highlight FARM
        farm_idx = method_order.index('farm') if 'farm' in method_order else -1
        if farm_idx >= 0:
            bars[farm_idx].set_edgecolor('#2c3e50')
            bars[farm_idx].set_linewidth(2.5)
            bars[farm_idx].set_hatch('//')

        ax.set_xticks(range(len(method_order)))
        # Rotate labels 45 degrees with right-alignment — prevents any overlap
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        # Add bottom margin so rotated labels don't get clipped
        ax.margins(x=0.08)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontweight='bold', pad=10, fontsize=11)
        ax.axhline(0, color='black', linewidth=0.5, linestyle='-')
        ax.grid(axis='y', alpha=0.3, zorder=0)

        # Value labels — place above/below bar with enough clearance
        for i, (bar, val) in enumerate(zip(bars, vals)):
            if val >= 0:
                ypos = val + 0.005
                va = 'bottom'
            else:
                ypos = val - 0.006
                va = 'top'
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f'{val:.3f}', ha='center', va=va,
                    fontsize=8, fontweight='bold' if i == farm_idx else 'normal')

    fig.suptitle('FARM vs. Baselines: Continual Learning on 5 Diverse Tasks\n'
                 '(Mistral-7B-Instruct-v0.3, Sequential Training T1→T5)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout(pad=1.5)
    plt.savefig(f"{OUT_DIR}/fig1_main_results.pdf")
    plt.savefig(f"{OUT_DIR}/fig1_main_results.png")
    plt.close()
    print("✓ Figure 1: Main results bar chart")


# ── Figure 2: ACC vs BWT Scatter (Pareto frontier) ───────────────────────────
def fig2_pareto_scatter():
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Compute smart label offsets to avoid overlaps
    # Custom offsets per method to prevent any label collision
    label_offsets = {
        'full_finetune': (0.001, -0.013),
        'ewc':           (0.001,  0.005),
        'o_lora':        (0.001,  0.005),
        'lora_replay':   (0.001,  0.005),
        'magmax':        (0.001, -0.013),
        'codyre':        (-0.006, -0.013),
        'farm':          (-0.012,  0.007),
    }

    for method_key, label in METHODS.items():
        if method_key not in results:
            continue
        m = results[method_key]['final_cl_metrics']
        acc = m.get('acc', 0)
        bwt = m.get('bwt', 0)
        size = 250 if method_key == 'farm' else 130
        zorder = 5 if method_key == 'farm' else 3

        ax.scatter(bwt, acc,
                   color=COLORS[method_key],
                   marker=MARKERS[method_key],
                   s=size, zorder=zorder,
                   edgecolors='black' if method_key == 'farm' else 'white',
                   linewidths=1.5 if method_key == 'farm' else 0.5,
                   label=label)

        dx, dy = label_offsets.get(method_key, (0.001, 0.004))
        ax.annotate(label, (bwt, acc),
                    xytext=(bwt + dx, acc + dy),
                    fontsize=8.5,
                    fontweight='bold' if method_key == 'farm' else 'normal',
                    color=COLORS[method_key])

    # Directional arrows for "ideal" corner — placed away from data points
    ax.annotate('', xy=(-0.001, 0.405), xytext=(-0.001, 0.375),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(-0.001, 0.372, 'Better ACC', ha='center', va='top',
            fontsize=8, color='gray', style='italic')
    ax.annotate('', xy=(-0.002, 0.360), xytext=(-0.055, 0.360),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
    ax.text(-0.057, 0.360, 'Less\nForgetting', ha='right', va='center',
            fontsize=8, color='gray', style='italic')

    ax.set_xlabel('Backward Transfer (BWT) ↑\n(higher = less catastrophic forgetting)', fontsize=11)
    ax.set_ylabel('Average Accuracy (ACC) ↑', fontsize=11)
    ax.set_title('ACC vs. Forgetting Trade-off\nIdeal: upper-right corner',
                 fontweight='bold', pad=10)
    ax.grid(alpha=0.25)
    ax.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_pareto_scatter.pdf")
    plt.savefig(f"{OUT_DIR}/fig2_pareto_scatter.png")
    plt.close()
    print("✓ Figure 2: ACC vs BWT Pareto scatter")


# ── Figure 3: Performance Matrix Heatmaps ────────────────────────────────────
def fig3_perf_matrices():
    selected = ['full_finetune', 'ewc', 'o_lora', 'farm']
    available = [m for m in selected if m in results]

    # Wider figure, larger fonts throughout
    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 5.5))
    if len(available) == 1:
        axes = [axes]

    task_labels = ['XSum', 'CNN/DM', 'MedQA', 'GSM8K', 'H.Eval']

    for ax, method_key in zip(axes, available):
        pm = results[method_key].get('performance_matrix', [])
        if not pm:
            continue

        matrix = np.full((5, 5), np.nan)
        for i, row in enumerate(pm):
            for j, val in enumerate(row):
                if val is not None and i < 5 and j < 5:
                    matrix[i][j] = val

        # Mask upper triangle (future tasks not yet trained)
        mask = np.zeros_like(matrix, dtype=bool)
        for i in range(5):
            for j in range(5):
                if j > i:
                    mask[i][j] = True

        sns.heatmap(matrix, ax=ax, mask=mask,
                    cmap='RdYlGn', vmin=0.1, vmax=0.6,
                    annot=True, fmt='.3f',
                    annot_kws={'size': 11, 'weight': 'bold'},
                    xticklabels=task_labels, yticklabels=task_labels,
                    cbar=True, linewidths=0.8, linecolor='white',
                    cbar_kws={'shrink': 0.8, 'label': 'Score'})

        ax.set_title(f'{METHODS[method_key]}', fontweight='bold', pad=10, fontsize=13)
        ax.set_xlabel('Task Evaluated On', fontsize=11, labelpad=8)
        ax.set_ylabel('After Training Task', fontsize=11, labelpad=8)
        ax.tick_params(axis='x', rotation=30, labelsize=10)
        ax.tick_params(axis='y', rotation=0, labelsize=10)

        # Fix colorbar font size
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Score', fontsize=10)

    fig.suptitle('Performance Matrix R[i,j]: Score on Task j after Training on Task i\n'
                 '(Diagonal = current task; off-diagonal = retention of previous tasks)',
                 fontsize=12, fontweight='bold', y=1.03)
    plt.tight_layout(pad=1.5)
    plt.savefig(f"{OUT_DIR}/fig3_perf_matrices.pdf")
    plt.savefig(f"{OUT_DIR}/fig3_perf_matrices.png")
    plt.close()
    print("✓ Figure 3: Performance matrix heatmaps")


# ── Figure 4: Per-Task Retention Curves ──────────────────────────────────────
def fig4_retention_curves():
    key_methods = ['full_finetune', 'ewc', 'o_lora', 'farm']
    available = [m for m in key_methods if m in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: T1 (XSum) retention across tasks
    ax = axes[0]
    for method_key in available:
        pm = results[method_key].get('performance_matrix', [])
        if not pm:
            continue
        xsum_scores = []
        for i, row in enumerate(pm):
            if i < len(pm) and len(row) > 0 and row[0] is not None:
                xsum_scores.append(row[0])

        if xsum_scores:
            x = list(range(1, len(xsum_scores) + 1))
            ax.plot(x, xsum_scores,
                    color=COLORS[method_key],
                    marker=MARKERS[method_key],
                    linewidth=2.5 if method_key == 'farm' else 1.5,
                    markersize=8 if method_key == 'farm' else 6,
                    linestyle='-' if method_key == 'farm' else '--',
                    label=METHODS[method_key],
                    zorder=5 if method_key == 'farm' else 3)

    ax.set_xlabel('Number of Tasks Trained', fontsize=11)
    ax.set_ylabel('XSum ROUGE Score', fontsize=11)
    ax.set_title('T1 (XSum) Retention\nAs More Tasks Are Learned', fontweight='bold')
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels([f'After T{i}' for i in range(1, 6)], rotation=15, ha='right')
    ax.legend(loc='upper right', framealpha=0.9, fontsize=9)
    ax.grid(alpha=0.25)

    # Right: Final per-task scores comparison (FARM vs best baseline O-LoRA)
    ax = axes[1]
    task_names = ['XSum', 'CNN/DM', 'MedQA', 'GSM8K', 'HumanEval']
    x = np.arange(len(task_names))
    width = 0.35

    compare_methods = [m for m in ['o_lora', 'farm'] if m in results]
    offsets = [-width / 2, width / 2]

    for offset, method_key in zip(offsets, compare_methods):
        pm = results[method_key].get('performance_matrix', [])
        if not pm or len(pm) < 5:
            continue
        final_scores = []
        for i in range(5):
            if i < len(pm) and len(pm[i]) > i and pm[i][i] is not None:
                final_scores.append(pm[i][i])
            else:
                final_scores.append(0)

        bars = ax.bar(x + offset, final_scores, width,
                      label=METHODS[method_key],
                      color=COLORS[method_key],
                      edgecolor='white', linewidth=0.8,
                      hatch='//' if method_key == 'farm' else '')

        # Value labels on top of each bar
        for bar, val in zip(bars, final_scores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f'{val:.2f}',
                ha='center', va='bottom',
                fontsize=7.5, fontweight='bold',
                color='#1a1a1a'
            )

    ax.set_xlabel('Task', fontsize=11)
    ax.set_ylabel('Score (at time of training)', fontsize=11)
    ax.set_title('Per-Task Learning Performance\nFARM vs. Best Baseline (O-LoRA)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=10)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    plt.tight_layout(pad=1.5)
    plt.savefig(f"{OUT_DIR}/fig4_retention_curves.pdf")
    plt.savefig(f"{OUT_DIR}/fig4_retention_curves.png")
    plt.close()
    print("✓ Figure 4: Retention curves and per-task comparison")


# ── Figure 5: FWT Analysis ────────────────────────────────────────────────────
def fig5_fwt_analysis():
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    method_order = sorted(results.keys(),
                          key=lambda m: results[m]['final_cl_metrics'].get('fwt', 0),
                          reverse=True)

    fwt_vals = [results[m]['final_cl_metrics'].get('fwt', 0) for m in method_order]
    colors = [COLORS[m] for m in method_order]
    # Use full names for y-axis (horizontal bar chart has plenty of room)
    labels = [METHODS[m] for m in method_order]

    bars = ax.barh(range(len(method_order)), fwt_vals, color=colors,
                   edgecolor='white', linewidth=0.8, zorder=3, height=0.6)

    farm_idx = method_order.index('farm') if 'farm' in method_order else -1
    if farm_idx >= 0:
        bars[farm_idx].set_edgecolor('#2c3e50')
        bars[farm_idx].set_linewidth(2)
        bars[farm_idx].set_hatch('//')

    ax.set_yticks(range(len(method_order)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Forward Transfer (FWT) ↑\n(higher = better zero-shot generalization to future tasks)',
                  fontsize=10)
    ax.set_title('Forward Transfer Comparison\nHow well does prior training help on new tasks?',
                 fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.25, zorder=0)

    # Value labels — always place to the RIGHT of bar end (outside bar)
    # This avoids any overlap with y-axis method names
    for i, (bar, val) in enumerate(zip(bars, fwt_vals)):
        xpos = val + 0.003 if val >= 0 else val - 0.003
        ha = 'left' if val >= 0 else 'right'
        ax.text(xpos, bar.get_y() + bar.get_height() / 2,
                f'{val:+.3f}',
                va='center', ha=ha,
                fontsize=9, color='black',
                fontweight='bold' if i == farm_idx else 'normal')

    # Extend x-axis to give labels room on both sides
    ax.set_xlim(left=ax.get_xlim()[0] - 0.025, right=ax.get_xlim()[1] + 0.020)
    plt.tight_layout(pad=1.2)
    plt.savefig(f"{OUT_DIR}/fig5_fwt_analysis.pdf")
    plt.savefig(f"{OUT_DIR}/fig5_fwt_analysis.png")
    plt.close()
    print("✓ Figure 5: FWT analysis")


# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"Generating figures from {len(results)} methods: {list(results.keys())}")
    fig1_main_results()
    fig2_pareto_scatter()
    fig3_perf_matrices()
    fig4_retention_curves()
    fig5_fwt_analysis()
    print(f"\nAll figures saved to {OUT_DIR}/")

"""
Generate the figures for the Apollo v1 vs v2 paper.

Outputs go to /home/claude/apollo_paper/figs/ as PDF (vector) for inclusion
in the LaTeX document. All figures use a consistent neutral palette to
match an academic-style document.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "figs"
OUT.mkdir(exist_ok=True)

# Neutral, print-safe palette.
C_V1 = "#3b3b3b"   # dark grey for v1
C_V2 = "#c44e1f"   # warm rust for v2
C_BASE = "#999999" # baseline grey

# Consistent fonts/sizes.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# -----------------------------------------------------------------------------
# Figure 1: Loss reduction normalized to baseline.
# Argues that v2 is fairly compared to v1 only after subtracting the
# vocab-size-dependent random baseline.
# -----------------------------------------------------------------------------

def fig_loss_reduction():
    versions = ["v1", "v2"]
    baselines = [10.82, 10.37]  # ln(50260), ln(32000)
    bests = [3.67, 3.95]
    reductions = [b - v for b, v in zip(baselines, bests)]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    # Left: absolute losses bar
    ax = axes[0]
    x = np.arange(len(versions))
    w = 0.36
    ax.bar(x - w / 2, baselines, width=w, color=C_BASE, label=r"random baseline $\ln V$")
    ax.bar(x + w / 2, bests, width=w, color=[C_V1, C_V2], label="best val loss")
    for i, (b, v) in enumerate(zip(baselines, bests)):
        ax.text(i - w / 2, b + 0.15, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i + w / 2, v + 0.15, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("cross-entropy (nats / token)")
    ax.set_title("Absolute losses")
    ax.set_ylim(0, 12)
    ax.legend(loc="upper right", frameon=False)

    # Right: reduction bar
    ax = axes[1]
    bars = ax.bar(versions, reductions, color=[C_V1, C_V2])
    for i, r in enumerate(reductions):
        ax.text(i, r + 0.1, f"{r:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("loss reduction from baseline")
    ax.set_title("Reduction (baseline minus best)")
    ax.set_ylim(0, 8.5)

    fig.savefig(OUT / "fig1_loss_reduction.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: Tokens per byte for the two tokenizers.
# Helps explain why v2 trained on fewer total tokens despite the same raw
# corpus size.
# -----------------------------------------------------------------------------

def fig_tokens_per_byte():
    # v1 and v2 corpora are similar but not identical. We report tokens
    # produced per MB of raw text per register.
    registers = ["literature", "wiki", "code"]

    # v1 stats: 12.88M / 47.86MB lit, 12.27M / 52.44MB wiki, 24.05M / 52.46MB code
    v1_tpb = [12.88 / 47.86, 12.27 / 52.44, 24.05 / 52.46]

    # v2 stats: 12.50M / 47.86MB lit, 12.68M / 52.44MB wiki, 11.05M / 22.20MB code
    v2_tpb = [12.50 / 47.86, 12.68 / 52.44, 11.05 / 22.20]

    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    x = np.arange(len(registers))
    w = 0.36
    ax.bar(x - w / 2, v1_tpb, width=w, color=C_V1, label="v1 (GPT-2 BPE, V=50260)")
    ax.bar(x + w / 2, v2_tpb, width=w, color=C_V2, label="v2 (custom BPE, V=32000)")
    for i, (a, b) in enumerate(zip(v1_tpb, v2_tpb)):
        ax.text(i - w / 2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.005, f"{b:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(registers)
    ax.set_ylabel("tokens per byte (M / MB)")
    ax.set_title("Tokenization efficiency by register")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(OUT / "fig2_tokens_per_byte.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: Token mix by register. Shows v2 is much more balanced than v1.
# -----------------------------------------------------------------------------

def fig_token_mix():
    registers = ["literature", "wiki", "code"]
    v1 = [12.88, 12.27, 24.05]
    v2 = [12.50, 12.68, 11.05]
    v1_pct = [100 * t / sum(v1) for t in v1]
    v2_pct = [100 * t / sum(v2) for t in v2]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    # Left: stacked-bar percentage view.
    ax = axes[0]
    cmap = ["#7f7f7f", "#bdbdbd", C_V2]
    bottoms_v1 = np.cumsum([0] + v1_pct[:-1])
    bottoms_v2 = np.cumsum([0] + v2_pct[:-1])
    for i, reg in enumerate(registers):
        ax.bar(["v1"], [v1_pct[i]], bottom=bottoms_v1[i], color=cmap[i],
               edgecolor="white", linewidth=0.5)
        ax.bar(["v2"], [v2_pct[i]], bottom=bottoms_v2[i], color=cmap[i],
               edgecolor="white", linewidth=0.5,
               label=reg if True else None)
        # Inline labels
        ax.text(0, bottoms_v1[i] + v1_pct[i] / 2, f"{reg} {v1_pct[i]:.0f}%",
                ha="center", va="center", fontsize=8, color="white")
        ax.text(1, bottoms_v2[i] + v2_pct[i] / 2, f"{reg} {v2_pct[i]:.0f}%",
                ha="center", va="center", fontsize=8, color="white")
    ax.set_ylabel("% of training tokens")
    ax.set_title("Register share of training tokens")
    ax.set_ylim(0, 100)

    # Right: absolute token counts, grouped bar.
    ax = axes[1]
    x = np.arange(len(registers))
    w = 0.36
    ax.bar(x - w / 2, v1, width=w, color=C_V1, label="v1")
    ax.bar(x + w / 2, v2, width=w, color=C_V2, label="v2")
    for i, (a, b) in enumerate(zip(v1, v2)):
        ax.text(i - w / 2, a + 0.5, f"{a:.1f}M", ha="center", fontsize=8)
        ax.text(i + w / 2, b + 0.5, f"{b:.1f}M", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(registers)
    ax.set_ylabel("tokens (millions)")
    ax.set_title("Tokens per register")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, 28)

    fig.savefig(OUT / "fig3_token_mix.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 4: Body params vs total params under different vocab sizes.
# Argues that the v1 -> v2 total-param drop comes from the embedding table,
# not the transformer body.
# -----------------------------------------------------------------------------

def fig_param_decomposition():
    versions = ["v1\n(V=50260)", "v2\n(V=32000)"]
    body = [25.31, 25.31]
    embd = [19.30, 10.24]
    other = [51.05 - 25.31 - 19.30, 41.70 - 25.31 - 10.24]  # leftover, basically 0

    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    x = np.arange(len(versions))
    ax.bar(x, body, color="#404040", label="transformer body (8L $\\times$ 8H $\\times$ 512D)")
    ax.bar(x, embd, bottom=body, color=C_V2, alpha=0.85,
           label="token embedding ($V \\times D$)")
    ax.bar(x, other, bottom=[a + b for a, b in zip(body, embd)],
           color="#cccccc", label="LayerNorm + position embedding")
    for i, (b, e) in enumerate(zip(body, embd)):
        ax.text(i, b / 2, f"{b:.2f}M", ha="center", va="center",
                fontsize=9, color="white")
        ax.text(i, b + e / 2, f"{e:.2f}M", ha="center", va="center",
                fontsize=9, color="white")
        ax.text(i, b + e + other[i] + 1.5, f"total\n{b + e + other[i]:.2f}M",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(versions)
    ax.set_ylabel("parameters (millions)")
    ax.set_title("Parameter decomposition: body vs embedding")
    ax.set_ylim(0, 70)
    ax.legend(loc="upper center", frameon=False, ncol=1,
              bbox_to_anchor=(0.5, -0.20))
    fig.savefig(OUT / "fig4_param_decomposition.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 5: Failure-mode comparison matrix. A 2x3 yes/no/changed grid showing
# the qualitative outcomes of the OOD stress tests.
# -----------------------------------------------------------------------------

def fig_failure_matrix():
    rows = ["literature OOD\n(spaceship prompt)",
            "wiki OOD\n(quantum entanglement)",
            "code OOD\n(NeuralNetwork class)"]
    cols = ["v1 outcome", "v2 outcome"]

    # 0=fail, 1=partial, 2=pass; we use color + text
    cells = [
        # literature OOD
        ["fail: collapses to\n[Illustration] loop",
         "improved: stays in\nliterary register"],
        # wiki OOD
        ["fail: ungrammatical\nFrench output",
         "changed: English,\nover-confident sports bio"],
        # code OOD
        ["fail: empty completion,\nthen test-fixture stubs",
         "improved: production-style\ntype-hinted code"],
    ]
    statuses = [
        [0, 2],
        [0, 1],
        [0, 2],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 3)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(False)

    cell_colors = {0: "#f3d6cb", 1: "#fae8c8", 2: "#d4ecd4"}
    border_colors = {0: "#a14225", 1: "#a37c1f", 2: "#3a7d3a"}

    # Header row above the cells.
    for j, c in enumerate(cols):
        ax.text(j + 0.5, 3.05, c, ha="center", va="bottom",
                fontsize=10, weight="bold")

    for i, row_label in enumerate(rows):
        # Row label (placed to the LEFT of the matrix).
        y_top = 3 - i  # row i occupies y in [y_top - 1, y_top]
        ax.text(-0.05, y_top - 0.5, row_label, ha="right", va="center",
                fontsize=9)
        for j in range(2):
            x = j
            y = y_top - 1
            color = cell_colors[statuses[i][j]]
            border = border_colors[statuses[i][j]]
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color,
                                        edgecolor=border, linewidth=1.2))
            ax.text(x + 0.5, y + 0.5, cells[i][j], ha="center", va="center",
                    fontsize=8.5)

    # Title above the header row, well clear of it.
    ax.text(1.0, 3.7, "OOD stress-test outcomes",
            ha="center", va="bottom", fontsize=11, weight="bold")
    ax.set_ylim(0, 4.0)
    ax.set_xlim(-1.4, 2.05)
    fig.savefig(OUT / "fig5_failure_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 6: Cosine LR schedule + hypothetical val curve.
# Illustrates the early-stopping mechanism added in v2.
# -----------------------------------------------------------------------------

def fig_schedule():
    iters = np.arange(0, 8001, 50)
    warmup = 320
    decay_to = 8000
    base_lr = 2.5e-4
    min_lr = 2.5e-5

    def lr_fn(it):
        if it < warmup:
            return base_lr * (it + 1) / warmup
        if it >= decay_to:
            return min_lr
        ratio = (it - warmup) / (decay_to - warmup)
        coeff = 0.5 * (1.0 + np.cos(np.pi * ratio))
        return min_lr + coeff * (base_lr - min_lr)

    lrs = np.array([lr_fn(i) for i in iters])

    # Mock illustrative curves to show v1's late overfit vs v2's plateau.
    # These are NOT the actual loss curves - we don't have them logged at
    # 50-iter granularity. They illustrate the conceptual story only.
    np.random.seed(0)
    base = 10.5 - 6.5 * (1 - np.exp(-iters / 1500))
    v1_curve = base + 0.05 * np.random.randn(len(iters))
    # Add overfitting at the tail.
    overfit_bump = np.where(iters > 5500, (iters - 5500) / 1200.0, 0)
    v1_curve = v1_curve + 0.4 * overfit_bump
    # v2: similar shape, plateau at end (early stop fires).
    v2_curve = base + 0.05 * np.random.randn(len(iters)) + 0.25  # offset higher
    plateau_idx = int(0.85 * len(iters))
    v2_curve[plateau_idx:] = v2_curve[plateau_idx]
    early_stop_iter = iters[plateau_idx + 8]  # patience=8

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    ax = axes[0]
    ax.plot(iters, lrs, color="#1c4e8a")
    ax.axvspan(0, warmup, color="#1c4e8a", alpha=0.10, label="warmup")
    ax.set_xlabel("iteration")
    ax.set_ylabel("learning rate")
    ax.set_title("Cosine LR schedule (warmup + decay)")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[1]
    ax.plot(iters, v1_curve, color=C_V1, label="v1: cosine to end (overfits)")
    ax.plot(iters, v2_curve, color=C_V2, label="v2: stops on val plateau")
    ax.axvline(early_stop_iter, color=C_V2, linestyle=":", alpha=0.7)
    ax.text(early_stop_iter - 200, v2_curve[plateau_idx] + 0.6, "early stop",
            fontsize=8, color=C_V2, ha="right")
    ax.set_xlabel("iteration")
    ax.set_ylabel("val loss (illustrative)")
    ax.set_title("Why early stopping was added")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_ylim(3, 11)

    fig.savefig(OUT / "fig6_schedule.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    fig_loss_reduction()
    fig_tokens_per_byte()
    fig_token_mix()
    fig_param_decomposition()
    fig_failure_matrix()
    fig_schedule()
    print("Wrote figures to", OUT)
    for p in sorted(OUT.iterdir()):
        print(" ", p.name, p.stat().st_size, "bytes")


if __name__ == "__main__":
    main()

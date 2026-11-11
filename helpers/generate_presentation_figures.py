"""
generate_presentation_figures.py
Generates clean, publication-quality figures for the AWQ academic presentation.
Saves PDF and PNG vector graphics into the figures/ directory.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer

from awq_lab import (
    quantize_weight_group,
    find_optimal_awq_scale,
    ActivationProfiler
)

# Set clean, rigorous academic style (pure white background, crisp dark lines, single crimson accent)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.titlesize": 13,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.color": "#e0e0e0",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(ROOT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Strict Academic Palette (like Figure 2 Layer 0 profiling)
C_DARK = "#111111"       # Primary text and lines
C_GRAY = "#555555"       # Secondary lines / baseline
C_LIGHT_GRAY = "#777777" # Unquantized / RTN bars
C_ACCENT = "#b71c1c"     # Single crimson accent for outliers & key findings
C_BORDER = "#cccccc"     # Crisp subtle border for callout boxes

# =============================================================================
# FIGURE 1: Roofline Analysis & Memory Bottleneck during Decoding
# =============================================================================
def plot_roofline():
    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#e5e5e5", zorder=0)
    
    # Apple Silicon M1 Unified Memory parameters:
    # Bandwidth: 68.25 GB/s, Peak FP16 compute: 5220 GFLOPS (5.22 TFLOPS)
    peak_bw = 68.25       # GB/s
    peak_compute = 5220.0 # GFLOPS (5.22 TFLOPS)
    I_knee = peak_compute / peak_bw # 76.47 FLOPs/Byte
    
    # Arithmetic intensities from 0.1 to 400 FLOP/Byte
    I = np.logspace(-1, 2.6, 200)
    P_mem = peak_bw * I
    P_attainable = np.minimum(P_mem, peak_compute)
    
    # Hardware Roofline Curve (clean dark line, publication standard)
    ax.plot(I, P_attainable, color=C_DARK, linewidth=2.4, zorder=3)
    
    # Dotted demarcation line at Knee
    ax.axvline(I_knee, color="#777777", linestyle=":", linewidth=1.1, zorder=2)
    # Knee label placed in dedicated whitespace at y=120 (far below flat ceiling and prefill box)
    ax.text(I_knee * 1.06, 120.0, f"Roofline Knee:\n{I_knee:.1f} FLOPs/B", fontsize=7.5, color="#555555", style="italic")
    
    # Operational Regime labels placed cleanly in dedicated non-interfering open areas
    # 1. Slanted Slope Regime: In open white space at upper left (above line)
    ax.text(0.13, 2900.0, "Memory-Bandwidth Bound\nRegime (Decoding Latency)", fontsize=7.8, color="#555555", style="italic")
    
    # 2. Flat Peak Regime: In open white space at lower right (below roofline)
    ax.text(110.0, 15.0, "Compute Bound Regime\n(Prefill Saturation)", fontsize=8.0, color="#555555", style="italic")
    
    # Operating Points: Each directly annotated with zero overlap or line collision
    
    # Point 1: FP16 Generation Stage (Intensity = 1.0 FLOP/Byte, Performance = 68.25 GFLOPS)
    I_gen_fp16 = 1.0
    P_gen_fp16 = peak_bw * I_gen_fp16
    ax.scatter([I_gen_fp16], [P_gen_fp16], color=C_ACCENT, s=70, zorder=5)
    # Box moved away from the slanted curve into the wide-open left whitespace (balanced between y-axis & line)
    ax.annotate("Decoding (FP16)\n1.0 FLOP/B · 68 GFLOPS\n(Severe Memory Wall)",
                (I_gen_fp16, P_gen_fp16),
                textcoords="offset points", xytext=(-78, 42),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.1),
                fontsize=8.0, color=C_ACCENT, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=C_ACCENT, lw=0.9))
    
    # Point 2: AWQ W4A16 Generation Stage (Intensity = 4.0 FLOPs/Byte, Performance = 273 GFLOPS)
    I_gen_w4 = 4.0
    P_gen_w4 = peak_bw * I_gen_w4
    ax.scatter([I_gen_w4], [P_gen_w4], color=C_DARK, s=80, zorder=5)
    # Box moved down and left into open whitespace, completely clearing the Knee label & graph line
    ax.annotate("Decoding W4A16 (AWQ)\n4.0 FLOPs/B · 273 GFLOPS\n(4x Throughput Speedup)",
                (I_gen_w4, P_gen_w4),
                textcoords="offset points", xytext=(8, -50),
                arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.1),
                fontsize=8.0, color=C_DARK, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=C_DARK, lw=0.9))
    
    # Point 3: Prefill Stage (Intensity = 160 FLOPs/Byte, Performance = 5220 GFLOPS)
    I_ctx = 160.0
    P_ctx = peak_compute
    ax.scatter([I_ctx], [P_ctx], color="#333333", s=70, zorder=5)
    # Box placed with centered alignment below flat roofline, 50pt above Knee label with zero collision
    ax.annotate("Prefill Stage (Context)\nCompute Bound: 5.2 TFLOPS",
                (I_ctx, P_ctx),
                textcoords="offset points", xytext=(-10, -42), ha="center",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1),
                fontsize=8.0, color=C_DARK,
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=C_BORDER, lw=0.8))
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 400.0)
    ax.set_ylim(6.0, 9500.0)
    ax.set_xlabel("Arithmetic Intensity (FLOPs / Byte)")
    ax.set_ylabel("Attainable Performance (GFLOPS)")
    ax.set_title("Roofline Analysis: Weight-Only Quantization Breaks the Memory Wall", pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roofline_memory_wall.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "roofline_memory_wall.png"))
    plt.close()
    print("-> Generated figures/roofline_memory_wall.pdf and .png (academic clean)")


# =============================================================================
# FIGURE 2: Activation Outlier Distribution & Channel Magnitudes
# =============================================================================
def plot_activation_outliers():
    model_id = "HuggingFaceTB/SmolLM-135M"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    
    linear = model.model.layers[0].mlp.gate_proj
    calibration_data = [
        "Large language models have revolutionized natural language processing and artificial intelligence.",
        "Quantum mechanics and general relativity are the two fundamental pillars of modern physics.",
        "Machine learning algorithms optimize mathematical loss functions over high-dimensional parameter spaces.",
        "Photosynthesis allows green plants to convert solar radiation into chemical energy stored in glucose."
    ]
    
    acts = []
    def hook(m, inp, out):
        acts.append(inp[0].detach().cpu())
        
    h = linear.register_forward_hook(hook)
    with torch.no_grad():
        for text in calibration_data:
            inputs = tokenizer(text, return_tensors="pt")
            model(**inputs)
    h.remove()
    
    X = torch.cat(acts, dim=1).squeeze(0) # [total_tokens, 576]
    s_X = X.abs().mean(dim=0).numpy()      # [576]
    
    mean_val = np.mean(s_X) # 0.0521
    top_indices = np.argsort(s_X)[-5:] # Top 5 outliers
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2), dpi=300, gridspec_kw={'width_ratios': [2.2, 1]})
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#e5e5e5", zorder=0)
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#e5e5e5", zorder=0)
    
    # Subplot 1: All channels
    channels = np.arange(len(s_X))
    ax1.plot(channels, s_X, color=C_GRAY, linewidth=1.0, alpha=0.9, label=r"Channel Saliency $s_X$")
    ax1.axhline(mean_val, color="#000000", linestyle="--", linewidth=1.2, label=f"Mean ({mean_val:.4f})")
    
    # Highlight top outliers
    ax1.scatter(top_indices, s_X[top_indices], color=C_ACCENT, s=40, zorder=5, label=r"Outliers ($>15\times$ Mean)")
    top_c = top_indices[-1]
    ax1.vlines(x=top_c, ymin=0, ymax=s_X[top_c], color=C_ACCENT, linestyle=":", linewidth=1.1)
    ax1.annotate(f"Channel {top_c}: {s_X[top_c]:.2f}\n({s_X[top_c]/mean_val:.1f}x Mean Outlier)",
                 (top_c, s_X[top_c]),
                 textcoords="offset points", xytext=(35, -20),
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.0),
                 fontsize=8.0, color=C_ACCENT, fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor=C_BORDER, lw=0.8))
    
    ax1.set_ylim(-0.02, 1.15)
    ax1.set_xlabel("Channel Index (Layer 0 MLP Input)")
    ax1.set_ylabel(r"Activation Magnitude $s_X$")
    ax1.set_title("Layer 0 Activation Channel Magnitudes", pad=8)
    ax1.legend(frameon=True, facecolor="white", edgecolor=C_BORDER, loc="upper left", fontsize=8.0)
    
    # Subplot 2: Sorted distribution (Long tail)
    sorted_s_X = np.sort(s_X)[::-1]
    sorted_pct = np.linspace(0, 100, len(sorted_s_X))
    ax2.plot(sorted_pct, sorted_s_X, color=C_ACCENT, linewidth=1.5)
    ax2.axvline(1.0, color="#333333", linestyle=":", linewidth=1.0)
    ax2.set_xlim(-2, 102)
    ax2.set_xlabel("Percentile of Channels (%)")
    ax2.set_ylabel("Magnitude")
    ax2.set_title("Sorted Saliency Profile", pad=8)
    ax2.text(14, sorted_s_X[4], "Top 1% Dominates\nSignal Energy", fontsize=8.0, color=C_ACCENT,
             bbox=dict(boxstyle="square,pad=0.3", facecolor="white", edgecolor=C_BORDER, lw=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "activation_outliers.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "activation_outliers.png"))
    plt.close()
    print("-> Generated figures/activation_outliers.pdf and .png (academic clean)")
    
    return linear.weight.data, X


# =============================================================================
# FIGURE 3: Alpha Grid Search vs Reconstruction MSE
# =============================================================================
def plot_alpha_grid_search(W, X):
    s_X = X.abs().mean(dim=0).clamp(min=1e-5)
    s_norm = s_X / s_X.mean()
    
    Y_true = X @ W.T
    
    alphas = np.linspace(0.0, 1.0, 21)
    mses = []
    
    for alpha in alphas:
        s = s_norm.pow(alpha)
        W_scaled = W * s.unsqueeze(0)
        W_scaled_q = quantize_weight_group(W_scaled, n_bits=4, group_size=64)
        W_awq = W_scaled_q / s.unsqueeze(0)
        mse = (Y_true - X @ W_awq.T).pow(2).mean().item()
        mses.append(mse * 1000.0) # Scale to 1e-3
        
    mses = np.array(mses)
    best_idx = np.argmin(mses)
    best_alpha = alphas[best_idx]
    best_mse = mses[best_idx]
    rtn_mse = mses[0]
    
    fig, ax = plt.subplots(figsize=(6.5, 3.4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, color="#e5e5e5", zorder=0)
    
    # Optimization Curve
    ax.plot(alphas, mses, marker='o', markersize=4.0, color=C_DARK, linewidth=1.8, zorder=3)
    ax.axhline(rtn_mse, color="#888888", linestyle="--", linewidth=1.2, zorder=2)
    ax.text(0.02, rtn_mse + 0.22, f"Naive RTN Baseline (alpha=0.0): MSE = {rtn_mse:.3f}e-3", fontsize=7.8, color="#555555", style="italic")
    
    # Highlight optimum with clean vertical drop line
    ax.scatter([best_alpha], [best_mse], color=C_ACCENT, s=85, zorder=5)
    ax.vlines(x=best_alpha, ymin=1.0, ymax=best_mse, color=C_ACCENT, linestyle=":", linewidth=1.1, alpha=0.8)
    reduction = (1.0 - best_mse / rtn_mse) * 100.0
    
    # Optimal alpha callout in open upper-left whitespace
    ax.annotate(f"Optimal $\\alpha^* = {best_alpha:.2f}$\nMSE: {best_mse:.3f}e-3\n(-{reduction:.1f}% Layer Error)",
                xy=(best_alpha, best_mse),
                xytext=(0.08, 4.8), textcoords="data",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.1),
                fontsize=8.5, color=C_ACCENT, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=C_ACCENT, lw=0.9))
    
    # Over-scaling penalty in open upper-center whitespace
    target_alpha = 0.75
    ax.annotate("Over-scaling penalty:\n$\\Delta$ step size explodes\nfor non-salient weights",
                xy=(target_alpha, mses[15]),
                xytext=(0.42, 6.8), textcoords="data",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0),
                fontsize=8.0, color="#333333",
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=C_BORDER, lw=0.8))
    
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(1.2, 9.0)
    ax.set_xlabel("AWQ Scaling Parameter $\\alpha$ (in $s = s_X^\\alpha$)")
    ax.set_ylabel("Output Reconstruction MSE ($\\times 10^{-3}$)")
    ax.set_title("Reconstruction Error vs Scaling Factor $\\alpha$ (INT4-g64)", pad=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "alpha_grid_search.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "alpha_grid_search.png"))
    plt.close()
    print("-> Generated figures/alpha_grid_search.pdf and .png (academic clean)")


# =============================================================================
# FIGURE 4: Logit Faithfulness & KL Divergence Comparison
# =============================================================================
def plot_logit_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.1), dpi=300)
    ax1.set_facecolor("white")
    ax2.set_facecolor("white")
    fig.patch.set_facecolor("white")
    ax1.grid(True, linestyle="--", linewidth=0.5, color="#e5e5e5", zorder=0, axis="y")
    ax2.grid(True, linestyle="--", linewidth=0.5, color="#e5e5e5", zorder=0, axis="y")
    
    # Data from results_summary.json
    models = ["Naive RTN", "AWQ (Ours)"]
    cos_sim = [0.99231, 0.99908]
    kl_div = [0.01052, 0.00198]
    
    colors = [C_LIGHT_GRAY, C_DARK]
    
    # Subplot 1: Cosine Similarity
    bars1 = ax1.bar(models, cos_sim, color=colors, width=0.38, edgecolor="#333333", linewidth=0.8, zorder=3)
    ax1.axhline(1.0000, color="#888888", linestyle=":", linewidth=0.9, zorder=2)
    ax1.set_ylim(0.988, 1.0022)
    ax1.set_ylabel(r"Cosine Similarity to FP32 ($\uparrow$)")
    ax1.set_title("Logit Directional Faithfulness", pad=10)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0004, f"{yval:.5f}", ha='center', va='bottom', fontsize=8.0, fontweight="bold")
        
    # Subplot 2: KL Divergence
    bars2 = ax2.bar(models, kl_div, color=colors, width=0.38, edgecolor="#333333", linewidth=0.8, zorder=3)
    ax2.set_ylim(0.0, 0.0148)
    ax2.set_ylabel(r"KL Divergence to FP32 ($\downarrow$)")
    ax2.set_title("Logit Distribution Shift", pad=10)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.0004, f"{yval:.5f}", ha='center', va='bottom', fontsize=8.0, fontweight="bold")
        
    # Callout badge placed cleanly in open upper whitespace above AWQ bar with clear downward arrow
    ax2.annotate("5.3x Lower\nDivergence\n(-81.2%)",
                xy=(1.0, 0.0028), xytext=(0.70, 0.0090),
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2, connectionstyle="arc3,rad=-0.12"),
                ha="center", va="center",
                fontsize=7.8, color=C_ACCENT, fontweight="bold",
                bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor=C_ACCENT, lw=0.9))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "logit_comparison.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "logit_comparison.png"))
    plt.close()
    print("-> Generated figures/logit_comparison.pdf and .png (academic clean)")


# =============================================================================
# FIGURE 5: AWQ Scaling & Operator Fusion Architectural Workflow
# =============================================================================
def plot_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Clean academic styling boxes (white fill, crisp dark borders)
    bbox_std = dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#555555", linewidth=1.0)
    bbox_awq = dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#111111", linewidth=1.0)
    bbox_awq_core = dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="#111111", linewidth=1.3)
    
    # Row 1: Traditional Linear Layer
    ax.text(0.02, 0.76, "Standard:", fontsize=10, fontweight="bold", color="#333333", va="center")
    ax.text(0.24, 0.76, "Input $X$\n(Activations)", bbox=bbox_std, ha="center", va="center", fontsize=8.5, color="#222222")
    ax.text(0.55, 0.76, "Weight $W$\n(FP16 Matrix)", bbox=bbox_std, ha="center", va="center", fontsize=8.5, color="#222222")
    ax.text(0.86, 0.76, "Output $Y = W X$\n(Ground Truth)", bbox=bbox_std, ha="center", va="center", fontsize=8.5, color="#222222")
    
    # Non-overlapping Arrows for Row 1
    ax.annotate("", xy=(0.43, 0.76), xytext=(0.35, 0.76), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#555555", mutation_scale=12))
    ax.annotate("", xy=(0.74, 0.76), xytext=(0.66, 0.76), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#555555", mutation_scale=12))
    
    # Subtle Divider line
    ax.axhline(0.53, xmin=0.02, xmax=0.98, color="#cccccc", linestyle="--", linewidth=0.9)
    
    # Row 2: AWQ Activation-Aware Scaled Layer
    ax.text(0.02, 0.28, "AWQ:", fontsize=10, fontweight="bold", color="#111111", va="center")
    ax.text(0.24, 0.28, "Scaled Input\n$X' = S^{-1} X$\n(Fused in RMSNorm)", bbox=bbox_awq, ha="center", va="center", fontsize=8.5, color="#111111")
    ax.text(0.55, 0.28, "Scaled Weights\n$W' = W \\cdot S$\n(Uniform INT4)", bbox=bbox_awq_core, ha="center", va="center", fontsize=8.5, color="#111111")
    ax.text(0.86, 0.28, "Output $\\hat{Y} \\approx W X$\n(Lossless INT4)", bbox=bbox_awq, ha="center", va="center", fontsize=8.5, color="#111111")
    
    # Non-overlapping Arrows for Row 2
    ax.annotate("", xy=(0.43, 0.28), xytext=(0.35, 0.28), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#111111", mutation_scale=12))
    ax.annotate("", xy=(0.74, 0.28), xytext=(0.66, 0.28), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#111111", mutation_scale=12))
    
    # Mathematical Equivalence annotation at bottom
    ax.text(0.50, 0.06, r"Exact Mathematical Invariance:  $Y = (W \cdot S) \cdot (S^{-1} X) \equiv W X$",
            ha="center", fontsize=9, style="italic", color="#222222",
            bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="#cccccc", lw=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "awq_scaling_pipeline.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "awq_scaling_pipeline.png"))
    plt.close()
    print("-> Generated figures/awq_scaling_pipeline.pdf and .png (academic clean)")


def main():
    print("Generating academic figures for presentation with subtle colors...")
    plot_roofline()
    W, X = plot_activation_outliers()
    plot_alpha_grid_search(W, X)
    plot_logit_comparison()
    plot_pipeline_diagram()
    print("All figures successfully created in figures/ directory!")


if __name__ == "__main__":
    main()

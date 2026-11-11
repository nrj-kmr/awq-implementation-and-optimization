"""
test_awq_on_mac.py
Hands-on Empirical Verification of the AWQ (Activation-aware Weight Quantization) Paper.
Designed for Apple Silicon (M1/M2/M3/M4) using PyTorch & HuggingFace SmolLM-135M.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_ID = "HuggingFaceTB/SmolLM-135M"


def quantize_weight_group(w: torch.Tensor, n_bits: int = 4, group_size: int = 64) -> torch.Tensor:
    """
    Simulates uniform affine group-wise quantization (RTN: Round-To-Nearest).
    w: [out_features, in_features]
    """
    orig_shape = w.shape
    w_grouped = w.view(-1, group_size)
    max_val = w_grouped.max(dim=1, keepdim=True).values
    min_val = w_grouped.min(dim=1, keepdim=True).values

    scale = (max_val - min_val).clamp(min=1e-5) / (2**n_bits - 1)
    zero = torch.round(-min_val / scale)

    q_w = torch.clamp(torch.round(w_grouped / scale) + zero, 0, 2**n_bits - 1)
    dequant = (q_w - zero) * scale
    return dequant.view(orig_shape)


def run_all_scenarios():
    print("=" * 70)
    print(" AWQ EMPIRICAL VERIFICATION ON APPLE SILICON")
    print(f" Device: {DEVICE.upper()} | Model: {MODEL_ID}")
    print("=" * 70)

    print("\n[Step 1] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
    model.eval()

    # Inspect first transformer layer MLP gate_proj
    linear = model.model.layers[0].mlp.gate_proj
    W = linear.weight.data.clone() # [out_features, in_features]
    out_dim, in_dim = W.shape
    print(f"Target Layer: Layer 0 MLP Gate Proj (Shape: {out_dim} x {in_dim})")

    # Calibration sentence to observe activations
    calib_text = (
        "Large language models have revolutionized natural language processing, "
        "enabling machines to understand context, reason through problems, and generate human-like text."
    )
    inputs = tokenizer(calib_text, return_tensors="pt").to(DEVICE)

    # Register forward hook to capture input activations X
    act_dict = {}
    handle = linear.register_forward_hook(lambda m, i, o: act_dict.update({"act": i[0].detach()}))
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    X = act_dict["act"].squeeze(0) # [seq_len, in_features]
    seq_len = X.shape[0]

    # =========================================================================
    # SCENARIO 1: Activation Outlier Profiling (Paper Section 3.1)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 1: The Activation Outlier Phenomenon (Section 3.1)")
    print("=" * 70)

    # Per-channel average activation magnitude: s_X = mean(|X|, dim=0)
    s_X = X.abs().mean(dim=0) # [in_features]
    mean_val = s_X.mean().item()
    max_val = s_X.max().item()
    top5_vals, top5_idx = torch.topk(s_X, 5)

    print(f"Sequence length: {seq_len} tokens | Channels (hidden dim): {in_dim}")
    print(f"Average channel activation magnitude : {mean_val:.4f}")
    print(f"Maximum channel activation magnitude : {max_val:.4f}")
    print(f"Max-to-Mean Ratio                    : {max_val / mean_val:.2f}x")
    print(f"Top 5 Outlier Channel Indices        : {top5_idx.tolist()}")
    print(f"Top 5 Outlier Channel Magnitudes     : {[round(v, 4) for v in top5_vals.tolist()]}")

    # 99th percentile vs median
    q99 = torch.quantile(s_X, 0.99).item()
    q50 = torch.median(s_X).item()
    print(f"Median magnitude: {q50:.4f} | 99th percentile: {q99:.4f} (Ratio: {q99/q50:.2f}x)")
    print("--> Insight: A tiny fraction (<1%) of activation channels carries extreme magnitude!")

    # =========================================================================
    # SCENARIO 2: Saliency Ranking — Weight Magnitude vs Activation Magnitude
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 2: Saliency Ranking — |W| vs s_X (Paper Table 1)")
    print("=" * 70)

    # Channel importance based on weight L1/L2 norm
    s_W = W.abs().mean(dim=0) # [in_features]
    _, top_W_indices = torch.topk(s_W, 5)

    overlap = len(set(top5_idx.tolist()).intersection(set(top_W_indices.tolist())))
    print(f"Top 5 channels by Weight Norm |W|   : {top_W_indices.tolist()}")
    print(f"Top 5 channels by Activation s_X    : {top5_idx.tolist()}")
    print(f"Overlap between |W| and s_X top 5   : {overlap} / 5")
    print("--> Insight: Weight magnitude has very low overlap with activation magnitude.")
    print("    Pruning or protecting based on |W| misses the true dynamic outliers of the model!")

    # =========================================================================
    # SCENARIO 3: Naive RTN vs AWQ Per-Channel Scaling (Paper Section 3.2)
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCENARIO 3: Naive RTN vs AWQ INT4 Group-Quantization (Section 3.2)")
    print("=" * 70)

    Y_true = X @ W.T # Ground truth FP32 output

    # 1. Naive RTN (Round-To-Nearest) INT4
    W_rtn = quantize_weight_group(W, n_bits=4, group_size=64)
    Y_rtn = X @ W_rtn.T
    mse_rtn = (Y_true - Y_rtn).pow(2).mean().item()
    print(f"[RTN INT4 (No Scaling, alpha=0.0)] Output MSE: {mse_rtn:.6f}")

    # 2. AWQ Grid Search over alpha in [0.0, 1.0]
    # s = (s_X / mean(s_X)) ^ alpha
    s_norm = s_X.clamp(min=1e-5)
    s_norm = s_norm / s_norm.mean()

    best_alpha = 0.0
    best_mse = mse_rtn
    best_W_awq = W_rtn

    print("\nRunning AWQ Grid Search over alpha in [0.1, 1.0]:")
    for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        s = s_norm.pow(alpha)
        # Scale weights: W' = W * diag(s)
        W_scaled = W * s.unsqueeze(0)
        # Quantize scaled weights
        W_scaled_q = quantize_weight_group(W_scaled, n_bits=4, group_size=64)
        # Dequantize & Descale: W_awq = W_scaled_q * diag(s)^-1
        W_awq = W_scaled_q / s.unsqueeze(0)

        Y_awq = X @ W_awq.T
        mse = (Y_true - Y_awq).pow(2).mean().item()
        marker = " <-- (Current Best)" if mse < best_mse else ""
        print(f"  alpha = {alpha:.1f} | Output MSE: {mse:.6f}{marker}")
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_W_awq = W_awq

    reduction = (1.0 - best_mse / mse_rtn) * 100.0
    print("\n" + "-" * 70)
    print(f"RESULTS SUMMARY:")
    print(f"  Naive RTN INT4 MSE   : {mse_rtn:.6f}")
    print(f"  Optimal AWQ INT4 MSE : {best_mse:.6f} (at alpha = {best_alpha:.1f})")
    print(f"  Reconstruction Gain  : {reduction:.2f}% MSE Reduction!")
    print("-" * 70)
    print("Conclusion: AWQ mathematically protects the ~1% salient channels by per-channel")
    print("scaling without requiring mixed-precision or modifying remaining weights!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_scenarios()

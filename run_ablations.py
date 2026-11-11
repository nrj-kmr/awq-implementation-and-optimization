"""
run_ablations.py
Comprehensive Ablation Study: Bit-width (4-bit vs 3-bit vs 2-bit)
and Group Size (G=32 vs G=64 vs G=128) on Apple Silicon M1.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from awq_lab import quantize_weight_group, find_optimal_awq_scale

def run_ablation_study(model_id: str = "HuggingFaceTB/SmolLM-135M"):
    print("=" * 75)
    print(" AWQ ABLATION STUDY: BIT-WIDTH & GROUP SIZE SENSITIVITY")
    print(f" Target Model: {model_id} | Device: MPS/CPU")
    print("=" * 75)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)

    linear = model.model.layers[0].mlp.gate_proj
    W = linear.weight.data

    calib_text = (
        "Large language models have revolutionized natural language processing, "
        "enabling machines to understand context, reason through problems, and generate human-like text."
    )
    inputs = tokenizer(calib_text, return_tensors="pt").to(device)

    act_dict = {}
    h = linear.register_forward_hook(lambda m, inp, out: act_dict.update({"act": inp[0].detach()}))
    with torch.no_grad():
        model(**inputs)
    h.remove()

    X = act_dict["act"].squeeze(0)
    Y_true = X @ W.T

    results = []

    print(f"\n{'Bits':>5} | {'Group':>5} | {'RTN MSE':>10} | {'AWQ MSE':>10} | {'MSE Red (%)':>12} | {'Alpha*':>7} | {'Metadata Overhead':>18}")
    print("-" * 75)

    for bits in [4, 3, 2]:
        for g_size in [32, 64, 128]:
            W_rtn = quantize_weight_group(W, n_bits=bits, group_size=g_size)
            mse_rtn = (Y_true - X @ W_rtn.T).pow(2).mean().item()

            scale, alpha, hist = find_optimal_awq_scale(
                W=W, X=X, n_bits=bits, group_size=g_size, grid_steps=20
            )
            best_mse = min(hist.values())
            red_pct = (1.0 - best_mse / mse_rtn) * 100.0

            overhead_pct = (16.0 / (g_size * bits)) * 100.0

            res_entry = {
                "bits": bits,
                "group_size": g_size,
                "rtn_mse": mse_rtn,
                "awq_mse": best_mse,
                "mse_reduction_pct": red_pct,
                "optimal_alpha": alpha,
                "metadata_overhead_pct": overhead_pct
            }
            results.append(res_entry)

            print(f"{bits:5d} | {g_size:5d} | {mse_rtn:10.6f} | {best_mse:10.6f} | {red_pct:11.2f}% | {alpha:7.2f} | {overhead_pct:17.2f}%")

    import os
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("-" * 75)
    print(f"Ablation study complete! Results saved to {out_path}")
    print("=" * 75)

def main():
    run_ablation_study()


if __name__ == "__main__":
    main()

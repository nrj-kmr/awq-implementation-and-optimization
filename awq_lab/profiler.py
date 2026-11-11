"""
awq_lab/profiler.py
Activation Profiling and Saliency Analysis module.
"""

from typing import Dict, Tuple, List
import torch
import torch.nn as nn


class ActivationProfiler:
    """
    Hooks into transformer linear layers to cache forward activations
    and extract channel-wise saliency statistics.
    """
    def __init__(self, model: nn.Module):
        self.model = model
        self.cached_activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

    def register_layer(self, name: str, module: nn.Module):
        def hook_fn(m, inp, out):
            # Input activations: [batch_size, seq_len, in_features]
            x = inp[0].detach()
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            if name not in self.cached_activations:
                self.cached_activations[name] = x
            else:
                self.cached_activations[name] = torch.cat([self.cached_activations[name], x], dim=0)

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        self.cached_activations.clear()

    @staticmethod
    def compute_channel_stats(X: torch.Tensor) -> Dict[str, float]:
        """
        Computes distribution moments of per-channel activation magnitudes:
        s_X = (1/T) * sum_{t=1}^T |X_{t, :}|
        """
        s_X = X.abs().mean(dim=0)
        mean_val = s_X.mean().item()
        max_val = s_X.max().item()
        median_val = torch.median(s_X).item()
        p99_val = torch.quantile(s_X, 0.99).item()

        return {
            "mean": mean_val,
            "max": max_val,
            "median": median_val,
            "p99": p99_val,
            "max_to_mean_ratio": max_val / (mean_val + 1e-8),
            "p99_to_median_ratio": p99_val / (median_val + 1e-8)
        }

    @staticmethod
    def compare_weight_vs_activation_saliency(
        W: torch.Tensor,
        X: torch.Tensor,
        top_k: int = 10
    ) -> Dict[str, object]:
        """
        Empirically verifies Section 3.1 & Table 1:
        Compares top-K salient channels identified by |W| (weight L1-norm)
        versus top-K salient channels identified by s_X (activation magnitude).
        """
        s_X = X.abs().mean(dim=0)          # [in_features]
        s_W = W.abs().mean(dim=0)          # [in_features]

        top_X_idx = torch.topk(s_X, top_k).indices.tolist()
        top_W_idx = torch.topk(s_W, top_k).indices.tolist()

        overlap = len(set(top_X_idx).intersection(set(top_W_idx)))
        overlap_pct = (overlap / top_k) * 100.0

        return {
            "top_k": top_k,
            "top_X_channels": top_X_idx,
            "top_W_channels": top_W_idx,
            "overlap_count": overlap,
            "overlap_percentage": overlap_pct
        }

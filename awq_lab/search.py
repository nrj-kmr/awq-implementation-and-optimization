"""
awq_lab/search.py
AWQ Scaling Grid Search and Weight Clipping Optimizer.
"""

from typing import Tuple, Dict, List
import torch
from .quantizer import quantize_weight_group


def find_optimal_awq_scale(
    W: torch.Tensor,
    X: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 64,
    grid_steps: int = 20,
    symmetric: bool = False
) -> Tuple[torch.Tensor, float, Dict[str, float]]:
    """
    Implements Equations (4) and (5) from Section 3.2:
    s = s_X^alpha
    alpha* = argmin_alpha || Q(W * diag(s)) * (diag(s)^-1 * X) - W * X ||^2

    Returns:
        best_scale: Optimal per-channel scaling vector [in_features]
        best_alpha: Optimal alpha value in [0, 1]
        loss_history: Dict of alpha -> MSE reconstruction loss
    """
    # 1. Compute per-channel activation scale s_X
    s_X = X.abs().mean(dim=0).clamp(min=1e-5)
    # Normalize s_X so scale operates around unit mean
    s_norm = s_X / s_X.mean()

    # Ground-truth floating point output
    Y_true = X @ W.T

    # Baseline RTN loss (alpha = 0.0)
    W_rtn = quantize_weight_group(W, n_bits=n_bits, group_size=group_size, symmetric=symmetric)
    Y_rtn = X @ W_rtn.T
    rtn_mse = (Y_true - Y_rtn).pow(2).mean().item()

    best_mse = rtn_mse
    best_alpha = 0.0
    best_scale = torch.ones_like(s_norm)

    loss_history = {0.0: rtn_mse}

    # Search grid from 0.05 to 1.0
    alphas = [i / grid_steps for i in range(1, grid_steps + 1)]
    for alpha in alphas:
        s = s_norm.pow(alpha)
        # Coordinate scaling: W' = W * diag(s) (broadcast along rows)
        W_scaled = W * s.unsqueeze(0)
        # Quantize scaled weight
        W_scaled_q = quantize_weight_group(W_scaled, n_bits=n_bits, group_size=group_size, symmetric=symmetric)
        # De-scale: W_awq = Q(W') * diag(s)^-1
        W_awq = W_scaled_q / s.unsqueeze(0)

        Y_awq = X @ W_awq.T
        mse = (Y_true - Y_awq).pow(2).mean().item()
        loss_history[round(alpha, 3)] = mse

        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_scale = s

    return best_scale, best_alpha, loss_history


def apply_awq_to_weights(
    W: torch.Tensor,
    scale: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 64,
    symmetric: bool = False
) -> torch.Tensor:
    """
    Applies the optimal per-channel scale, performs group quantization,
    and returns dequantized weights in the original coordinate system.
    """
    W_scaled = W * scale.unsqueeze(0)
    W_scaled_q = quantize_weight_group(W_scaled, n_bits=n_bits, group_size=group_size, symmetric=symmetric)
    W_awq = W_scaled_q / scale.unsqueeze(0)
    return W_awq

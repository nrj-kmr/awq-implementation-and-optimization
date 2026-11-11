"""
awq_lab/quantizer.py
Core Quantization Primitives: Group-wise Uniform Affine and Symmetric Quantization.
"""

import torch


def quantize_weight_group(
    w: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 64,
    symmetric: bool = False
) -> torch.Tensor:
    """
    Simulates uniform group-wise quantization and dequantization (Round-To-Nearest).

    Args:
        w: Weight tensor of shape [out_features, in_features]
        n_bits: Target bit-width (e.g., 4 or 3)
        group_size: Number of contiguous elements sharing a scale and zero-point
        symmetric: If True, uses symmetric quantization [-2^(B-1), 2^(B-1)-1]
                   If False, uses asymmetric affine quantization [0, 2^B - 1]
    Returns:
        Dequantized floating-point tensor of identical shape [out_features, in_features]
    """
    orig_shape = w.shape
    w_grouped = w.view(-1, group_size)

    q_max = 2**n_bits - 1

    if symmetric:
        max_abs = w_grouped.abs().max(dim=1, keepdim=True).values.clamp(min=1e-5)
        scale = max_abs / (2**(n_bits - 1) - 1)
        q_w = torch.clamp(torch.round(w_grouped / scale), -(2**(n_bits - 1)), 2**(n_bits - 1) - 1)
        dequant = q_w * scale
    else:
        max_val = w_grouped.max(dim=1, keepdim=True).values
        min_val = w_grouped.min(dim=1, keepdim=True).values
        scale = (max_val - min_val).clamp(min=1e-5) / q_max
        zero = torch.round(-min_val / scale)
        q_w = torch.clamp(torch.round(w_grouped / scale) + zero, 0, q_max)
        dequant = (q_w - zero) * scale

    return dequant.view(orig_shape)


def pack_int4_weights(q_w: torch.Tensor) -> torch.Tensor:
    """
    Packs two 4-bit unsigned integers into a single 8-bit unsigned integer (uint8).
    Demonstrates SIMD memory packing logic used in TinyChat.
    """
    assert q_w.dtype in [torch.int8, torch.int32, torch.int64, torch.float32]
    q_w_int = q_w.to(torch.uint8).view(-1)
    assert q_w_int.numel() % 2 == 0, "Number of weights must be even for 4-bit packing"

    w_low = q_w_int[0::2] & 0x0F
    w_high = (q_w_int[1::2] & 0x0F) << 4
    packed = w_low | w_high
    return packed


def unpack_int4_weights(packed: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """
    Unpacks 8-bit uint8 tensor back into two 4-bit values using bitwise masking.
    Matches equation: W_low = P & 0x0F, W_high = (P >> 4) & 0x0F
    """
    w_low = packed & 0x0F
    w_high = (packed >> 4) & 0x0F
    unpacked = torch.stack([w_low, w_high], dim=-1).view(-1)
    return unpacked.view(orig_shape).to(torch.float32)

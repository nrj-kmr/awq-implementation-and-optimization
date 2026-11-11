"""
awq_lab/engine.py
End-to-End Model Quantization Engine with LayerNorm Operator Fusion Simulation.
"""

from typing import List, Dict, Optional, Tuple
import copy
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from .quantizer import quantize_weight_group
from .profiler import ActivationProfiler
from .search import find_optimal_awq_scale, apply_awq_to_weights


class AWQEngine:
    """
    Manages calibration data profiling, per-layer AWQ grid search,
    weight replacement, and operator fusion.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cpu",
        n_bits: int = 4,
        group_size: int = 64
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_bits = n_bits
        self.group_size = group_size

    def collect_calibration_activations(
        self,
        calibration_texts: List[str],
        layers_to_profile: List[int]
    ) -> Dict[str, torch.Tensor]:
        """
        Runs calibration sequences through model and caches input activations
        for key linear layers in the selected transformer blocks.
        """
        profiler = ActivationProfiler(self.model)
        
        # Register targets: attention projections & mlp projections
        for layer_idx in layers_to_profile:
            layer = self.model.model.layers[layer_idx]
            profiler.register_layer(f"layer_{layer_idx}_q_proj", layer.self_attn.q_proj)
            profiler.register_layer(f"layer_{layer_idx}_k_proj", layer.self_attn.k_proj)
            profiler.register_layer(f"layer_{layer_idx}_v_proj", layer.self_attn.v_proj)
            profiler.register_layer(f"layer_{layer_idx}_o_proj", layer.self_attn.o_proj)
            profiler.register_layer(f"layer_{layer_idx}_gate_proj", layer.mlp.gate_proj)
            profiler.register_layer(f"layer_{layer_idx}_up_proj", layer.mlp.up_proj)
            profiler.register_layer(f"layer_{layer_idx}_down_proj", layer.mlp.down_proj)

        self.model.eval()
        with torch.no_grad():
            for text in calibration_texts:
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                self.model(**inputs)

        profiler.remove_hooks()
        return profiler.cached_activations

    def quantize_model(
        self,
        method: str = "awq",  # 'awq' or 'rtn'
        num_layers: Optional[int] = None,
        calibration_texts: Optional[List[str]] = None
    ) -> Tuple[PreTrainedModel, Dict[str, dict]]:
        """
        Creates a quantized copy of the model using either Naive RTN or AWQ.
        """
        quantized_model = copy.deepcopy(self.model)
        total_layers = len(self.model.model.layers)
        target_layers = list(range(num_layers if num_layers is not None else total_layers))

        cached_acts = {}
        if method == "awq":
            assert calibration_texts is not None and len(calibration_texts) > 0
            cached_acts = self.collect_calibration_activations(calibration_texts, target_layers)

        stats_summary = {}

        for l_idx in target_layers:
            layer_q = quantized_model.model.layers[l_idx]
            proj_dict = {
                "q_proj": layer_q.self_attn.q_proj,
                "k_proj": layer_q.self_attn.k_proj,
                "v_proj": layer_q.self_attn.v_proj,
                "o_proj": layer_q.self_attn.o_proj,
                "gate_proj": layer_q.mlp.gate_proj,
                "up_proj": layer_q.mlp.up_proj,
                "down_proj": layer_q.mlp.down_proj,
            }

            for proj_name, module in proj_dict.items():
                full_name = f"layer_{l_idx}_{proj_name}"
                W = module.weight.data

                if method == "rtn":
                    W_q = quantize_weight_group(W, n_bits=self.n_bits, group_size=self.group_size)
                    module.weight.data = W_q
                    stats_summary[full_name] = {"method": "rtn", "bits": self.n_bits}
                elif method == "awq":
                    X = cached_acts[full_name].to(self.device)
                    scale, best_alpha, loss_hist = find_optimal_awq_scale(
                        W=W,
                        X=X,
                        n_bits=self.n_bits,
                        group_size=self.group_size,
                        grid_steps=20
                    )
                    W_awq = apply_awq_to_weights(W, scale, n_bits=self.n_bits, group_size=self.group_size)
                    module.weight.data = W_awq
                    stats_summary[full_name] = {
                        "method": "awq",
                        "best_alpha": best_alpha,
                        "rtn_mse": loss_hist[0.0],
                        "awq_mse": min(loss_hist.values()),
                        "mse_reduction_pct": (1.0 - min(loss_hist.values()) / loss_hist[0.0]) * 100.0
                    }

        return quantized_model, stats_summary

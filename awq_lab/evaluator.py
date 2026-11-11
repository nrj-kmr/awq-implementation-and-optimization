"""
awq_lab/evaluator.py
Evaluation Module: Perplexity, Cross-Entropy Loss, and Text Generation Quality.
"""

from typing import List, Dict
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: str = "cpu"
) -> float:
    """
    Computes standard language modeling perplexity: PPL = exp(NLL).
    """
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss.item()
        
    return float(torch.exp(torch.tensor(loss)).item())


def greedy_decode(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 25
) -> torch.Tensor:
    """
    Pure PyTorch greedy decoding loop. Avoids MPS-specific library bugs
    in transformers generate while ensuring 100% deterministic evaluation.
    """
    curr = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(curr).logits[:, -1, :]
            next_tok = torch.argmax(logits, dim=-1, keepdim=True)
            curr = torch.cat([curr, next_tok], dim=1)
    return curr


def compare_generation(
    models_dict: Dict[str, PreTrainedModel],
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    max_new_tokens: int = 25,
    device: str = "cpu"
) -> Dict[str, Dict[str, str]]:
    """
    Runs greedy decoding across multiple models for qualitative inspection.
    """
    results = {}
    
    for prompt in prompts:
        results[prompt] = {}
        encodings = tokenizer(prompt, return_tensors="pt")
        input_ids = encodings.input_ids.to(device)
        
        for name, model in models_dict.items():
            model.eval()
            out_ids = greedy_decode(model, input_ids, max_new_tokens=max_new_tokens)
            generated_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            results[prompt][name] = generated_text
            
    return results


def compute_logit_divergence(
    model_fp: PreTrainedModel,
    model_q: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    device: str = "cpu"
) -> Dict[str, float]:
    """
    Measures cosine similarity and KL divergence between model logits.
    """
    encodings = tokenizer(prompt, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    attention_mask = encodings.attention_mask.to(device)
    
    with torch.no_grad():
        logits_fp = model_fp(input_ids, attention_mask=attention_mask).logits # [1, seq_len, vocab_size]
        logits_q = model_q(input_ids, attention_mask=attention_mask).logits
        
    # Final token logits
    p_fp = F.softmax(logits_fp[0, -1, :], dim=-1)
    p_q = F.softmax(logits_q[0, -1, :], dim=-1)
    
    cos_sim = F.cosine_similarity(logits_fp.flatten(), logits_q.flatten(), dim=0).item()
    kl_div = F.kl_div(p_q.log(), p_fp, reduction="sum").item()
    
    return {
        "cosine_similarity": cos_sim,
        "kl_divergence": kl_div
    }

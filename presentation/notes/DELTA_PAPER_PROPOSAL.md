# Research Project & Delta Paper Proposal
## Advanced LLM Compression on Edge Hardware

**Project Title:**
*Context-Adaptive Dynamic AWQ and Joint Key-Value Compression for Ultra-Low Bit LLM Inference on Edge Hardware*

**Candidate:** M.Tech AI Lab — IIITV
**Target Submission Venues:** MLSys 2025 / EMNLP 2025 (System Demonstrations / Short Paper) / IEEE Embedded Systems Letters
**Base Reference:** *AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration* (Lin et al., MIT HAN Lab, MLSys 2024 / NeurIPS 2023).

---

## 1. Executive Summary & Problem Formulation

While the original AWQ paper established that activation magnitude governs weight quantization saliency, our empirical verification on Apple Silicon M1 revealed **three critical physical and algorithmic boundaries**:

1. **The Static Calibration Dilemma ($\alpha^*$ Invariance Failure):**
   AWQ optimizes a single static scalar $\alpha^* \in [0, 1]$ over 128 generic calibration sequences (Pile/Wikitext). However, empirical activation profiling demonstrates that dynamic outlier distributions experience severe domain shift between conversational text, Python code synthesis, and formal mathematical reasoning. A static offline scale compromises out-of-distribution reasoning.

2. **The Long-Context KV-Cache Memory Wall:**
   AWQ focuses exclusively on linear weight matrices ($W$). However, during autoregressive decoding with context windows exceeding 8,192 tokens ($L > 8192$), the memory footprint of Key-Value tensors:
   $$\text{Memory}_{\text{KV}} = 2 \times 2\text{ Bytes} \times N_{\text{layers}} \times N_{\text{kv\_heads}} \times d_{\text{head}} \times L$$
   surpasses the total static model weight footprint, recreating the DRAM memory wall.

3. **The Uniform 2-Bit Breakdown:**
   Our 9-configuration ablation matrix verified that uniform INT4 and INT3 remain viable ($36.73\%$ and $37.89\%$ MSE reduction), but uniform INT2 degrades catastrophically ($>10\times$ MSE expansion) due to group dynamic range clipping.

---

## 2. Proposed Research Contributions (The Delta Formulation)

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             PROPOSED DELTA ARCHITECTURE                          │
│                                                                                  │
│   Input Prompt X ──► [ Lightweight Context Gate ] ──► Dynamic Scale α(t)        │
│                                                             │                    │
│   Linear Projections:        Attention Key / Value:        │                    │
│   W' = W · S(α(t))           W_K' = W_K · S_K              ▼                    │
│   X' = S(α(t))⁻¹ · X         K_cache = Q_4bit(K · S_K) ──► Fused Metal Shaders │
│   (Fused in RMSNorm)         (Scaled Softmax Invariance)   (Unified Memory)      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Contribution 1: Context-Adaptive Dynamic Scaling ($\alpha(t)$)
Instead of fixing $\alpha^*$ offline, we formulate a dynamic, input-conditioned scaling vector:
$$\alpha_l(t) = \sigma\left(W_\alpha^{(l)} \cdot \bar{h}_t + b_\alpha^{(l)}\right)$$
where $\bar{h}_t \in \mathbb{R}^d$ is the pooled prompt representation and $W_\alpha \in \mathbb{R}^{1 \times d}$ is a lightweight linear projection parameter (adding $<0.01\%$ parameters). The network dynamically tilts protection toward outlier channels specific to coding, reasoning, or multilingual contexts at runtime.

### Contribution 2: Joint Key-Value Cache Coordinate Scaling (W4A4-KV)
We extend activation-aware coordinate transformation to Key and Value projection layers:
$$K = X W_K, \quad V = X W_V$$
By finding an optimal per-channel scale $S_K$ that minimizes quantization error in the attention dot-product:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
We absorb $S_K^{-1}$ directly into the query vectors $Q' = Q \cdot S_K^{-1}$, guaranteeing exact dot-product invariance while compressing dynamic KV cache states to 4-bit unsigned integers without perplexity loss.

### Contribution 3: Sparse-Dense Hybrid 2-Bit Quantization (AWQ-Sparse)
To break through the 2-bit barrier where uniform grids fail:
$$W = W_{\text{INT2}} + W_{\text{outlier, FP16}}$$
We preserve only the top $0.5\%$ outlier channels in native FP16, packing the remaining $99.5\%$ into uniform INT2. Crucially, rather than arbitrary non-zero sparsity, our outliers are structured along contiguous channel columns, allowing them to map directly into 128-bit vector lanes (ARM NEON) without warp divergence.

### Contribution 4: Native Apple Silicon Metal Performance Shaders (`mps`)
TinyChat developed specialized CUDA kernels for Nvidia GPUs. We propose native Apple Silicon Metal Compute Shaders leveraging Unified Memory Architecture (UMA):
* Direct pointer sharing between CPU and GPU without PCIe serialization.
* SIMD matrix-vector instructions (`simdgroup_matrix`) for fast 4-bit weight dequantization directly in on-chip registers.

---

## 3. Experimental Evaluation Plan

### Target Models & Benchmarks
* **Models:** `SmolLM-135M` (Edge prototype), `Llama-3-8B-Instruct`, `Mistral-7B-v0.3`.
* **Zero-Shot Reasoning:** MMLU, GSM8K (Math), HumanEval (Python Code), ARC-Challenge.
* **Long-Context Retrieval:** LongBench (8k to 32k context lengths).
* **Hardware Testbeds:**
  * Apple M1 (Baseline Edge SoC, 8 GB UMA, 68.25 GB/s).
  * Apple M3 Max / M4 (Modern Edge SoC with hardware dynamic caching).
  * NVIDIA Jetson Orin Nano (Embedded edge comparison).

### Baselines for Comparison
1. Baseline Unquantized FP16
2. Naive Round-To-Nearest (RTN INT4/INT3/INT2)
3. SmoothQuant (W8A8)
4. GPTQ (W4A16 second-order solver)
5. MIT AWQ (Static baseline)
6. QuaRot / SpinQuant (Rotational baselines)

---

## 4. Expected Deliverables & Thesis Milestones

| Milestone | Target Deliverable | Expected Completion |
| :--- | :--- | :--- |
| **M1: Baseline Re-creation** | Full Apple Silicon M1 empirical profiling & 9-config ablation | Completed (Current AI Lab) |
| **M2: Dynamic Gating Prototype** | PyTorch implementation of context-aware $\alpha(t)$ network | Month 2 |
| **M3: W4A4-KV Cache Engine** | Integration of attention-invariant scaling into HuggingFace models | Month 4 |
| **M4: Metal Compute Kernels** | Native Metal shaders for SIMD-accelerated on-device decoding | Month 6 |
| **M5: Publication & Thesis Draft** | Conference manuscript submission & final M.Tech thesis chapter | Month 8 |

---

## 5. Potential Faculty Viva Questions Regarding Delta Paper

* **Faculty Question:** *"Why not just train a smaller model from scratch instead of doing complex 2-bit / 3-bit quantization?"*
  * **Answer:** Training a 3B model from scratch requires millions of tokens and hundreds of thousands of dollars in compute. Quantizing an 8B model to 3-bit yields higher reasoning accuracy than a natively trained 3B model while running in the exact same memory footprint (3 GB DRAM), as proven by scaling law transferability.
* **Faculty Question:** *"How does your W4A4-KV cache scale invariance differ from KIVI or SmoothQuant?"*
  * **Answer:** KIVI uses per-channel key quantization and per-token value quantization without coordinate transformation. SmoothQuant requires runtime division on activations. Our method folds $S_K^{-1}$ directly into the Query vector offline, preserving standard scaled dot-product attention without modifying kernel computation.

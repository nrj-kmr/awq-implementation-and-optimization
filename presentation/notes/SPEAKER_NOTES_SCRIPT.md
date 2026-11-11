# AWQ Presentation Oral Defense Script & Speaker Notes

## Timed Slide-by-Slide Delivery Guide for M.Tech AI Lab Seminar

* **Total Target Presentation Time:** 11 to 13 minutes (Leaves 2-4 minutes for Faculty Q&A)
* **Target Audience:** Department Faculty, External Examiners, and Graduate Peers
* **Core Presentation Strategy:** Speak deliberately, avoid reading slide text verbatim, reference the visual matrices/diagrams directly, highlight your Apple Silicon M1 empirical numbers, and finish strong with your Delta Paper research roadmap.

---

### Slide 1: Title & Strategic Overview (0:00 – 0:45)

* **Visual Anchor:** 3-Pillar Banner at the bottom.
* **What to say:**
  > "Respected faculty and peers, good morning. Today, I am presenting our critical study, empirical verification, and research roadmap for the paper: **'AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration'**, published at MLSys 2024 by Ji Lin, Song Han, and colleagues at MIT HAN Lab.
  >
  > In this seminar, I will walk you through the core memory bottleneck of on-device LLM inference, dissect why conventional post-training quantization degrades at 4-bit, explain the exact coordinate scaling mechanism of AWQ, present our own empirical implementation and 9-configuration ablation study on Apple Silicon M1, and outline concrete research proposals for a Delta Paper."
* **Transition:** *"Let us begin by looking at the physical hardware bottleneck that governs edge LLM generation."*

---

### Slide 2: Problem Statement — The Memory Wall in LLM Decoding (0:45 – 2:00)

* **Visual Anchor:** Left: Arithmetic intensity derivation. Right: Figure `roofline_memory_wall.pdf`.
* **What to point at:** Point to the red dot at 1.0 FLOP/Byte deep in the memory-bound slope, then show the 4x shift to the black dot.
* **What to say:**
  > "The primary limitation for deploying Large Language Models on edge hardware—such as Apple Silicon or edge NPUs—is not computing capacity, but **memory bandwidth**.
  >
  > In interactive autoregressive decoding with batch size 1, tokens are generated sequentially. For every single token produced, the hardware must read the entire weight matrix of every transformer layer from off-chip DRAM to on-chip SRAM registers, just to compute a single matrix-vector product $y = W x$.
  >
  > As derived on the left, the arithmetic intensity is only **1.0 FLOP per byte** in FP16. On our Apple M1 hardware with 68.25 GB/s memory bandwidth and 5.2 TFLOPS compute, the roofline knee is at **76.5 FLOPs/byte**.
  >
  > Because we operate at 1.0 FLOP/byte, the compute cores spend over 90% of their cycles idle, waiting for DRAM data. Since weight traffic is roughly a thousand times larger than activation traffic during generation, compressing weights to 4-bit cuts memory transfer by $4\times$, shifting arithmetic intensity to 4.0 FLOPs/byte and directly quadrupling decoding throughput."
* **Transition:** *"Given this clear objective, why couldn't existing quantization methods solve this problem?"*

---

### Slide 3: Prior Art Comparison Matrix (2:00 – 3:15)

* **Visual Anchor:** 4-Column Structured Prior Art Matrix.
* **What to point at:** Walk down the 'Critical Failure' column row-by-row, contrasting prior art with AWQ's highlighted row.
* **What to say:**
  > "Prior quantization methods each hit fundamental roadblocks when applied to edge LLMs:
  >
  > 1. **QAT (Quantization-Aware Training):** Retrains weights using backpropagation. While accurate, it is computationally intractable for 7B to 70B models, requiring massive GPU clusters and access to the original pre-training datasets.
  > 2. **Naive RTN (Round-To-Nearest):** Directly rounds floating-point weights to integers. While it works at 8-bit, at 4-bit it causes catastrophic perplexity explosion—for instance, OPT-1.3B perplexity explodes from 14.6 to 119.
  > 3. **SmoothQuant:** Focuses on W8A8 quantization. However, it is tailored for compute-heavy server batching and does not solve the memory wall during single-batch edge decoding.
  > 4. **GPTQ:** Uses second-order Taylor expansion to invert the Hessian matrix and update unquantized weights. While effective on perplexity, it overfits to the calibration data, altering weights in ways that degrade coding, math, and general reasoning.
  >
  > **AWQ's distinct advantage:** It modifies zero weights, requires zero backpropagation, uses standard uniform INT4 SIMD, and preserves general reasoning capability."
* **Transition:** *"The core breakthrough of AWQ begins with a surprising discovery regarding weight importance."*

---

### Slide 4: Core Discovery — Activation Outliers Dictate Saliency (3:15 – 4:30)

* **Visual Anchor:** Left: Saliency Paradox Table. Right: Figure `activation_outliers.pdf`.
* **What to point at:** Point to Channel 303 in the table ($17.3\times$ mean) and its spike on the right figure.
* **What to say:**
  > "For decades, neural network pruning assumed that larger weights ($|W|$) are more important. The authors tested this hypothesis and discovered **The Saliency Paradox**: protecting weights based on $|W|$ yields perplexity no better than random weight selection!
  >
  > Instead, true weight saliency is strictly governed by **Activation Magnitude** ($s_X = \mathbb{E}[|X|]$). A tiny fraction—less than 1%—of hidden channels exhibits extreme activation spikes.
  >
  > In our empirical profiling on Apple M1 shown on the slide, Channel 303 reaches an activation magnitude of 0.8995—which is **17.26 times the layer mean**. Yet in weight magnitude, Channel 303 is completely unremarkable, ranking 28th!
  >
  > Across the layer, weight norm misses 5 of the top 10 true outlier channels. If you quantize Channel 303 naively, its huge activation magnifies the quantization noise by $17\times$, corrupting the output."
* **Transition:** *"How does AWQ protect these salient channels without introducing mixed-precision hardware overhead?"*

---

### Slide 5: The AWQ Solution — Coordinate Scaling & Operator Fusion (4:30 – 5:45)

* **Visual Anchor:** Top: Invariant equation. Middle: TikZ flow diagram. Bottom: Green Operator Fusion callout card.
* **What to point at:** Point to the exact cancellation $(W \cdot S)(S^{-1} X) \equiv W X$, then the green fusion box.
* **What to say:**
  > "Rather than using irregular mixed-precision formats that break hardware execution, AWQ applies an **Exact Invariant Coordinate Transformation**:
  > $$Y = W X = (W \cdot S)(S^{-1} X) \equiv W X$$
  >
  > By multiplying the salient weight columns by a per-channel scale $S > 1$, we expand their dynamic range relative to the quantization grid, effectively giving them higher precision. Simultaneously, we multiply the activations by $S^{-1}$.
  >
  > Crucially, as highlighted in the green box, **$S^{-1}$ is not computed at runtime**. It is pre-computed offline and folded directly into the gain parameter of the preceding RMSNorm layer: $\gamma_{\text{fused}} = \gamma \odot s^{-1}$.
  >
  > At runtime, the hardware executes a completely standard RMSNorm followed by a pure, uniform INT4 GEMM. There are **zero additional FLOPs and zero runtime latency overhead**."
* **Transition:** *"Let us examine the mathematical proof of why scaling suppresses quantization error."*

---

### Slide 6: Theoretical Mechanism & Error Proof (5:45 – 7:15)

* **Visual Anchor:** Left Box: Quantization noise model. Right Box: 1D grid search optimization.
* **What to point at:** Point to $\hat{w} = w + \frac{\epsilon'}{s_c}$ and the $\frac{1}{s_c}$ reduction.
* **What to say:**
  > "To understand the mathematical mechanism:
  >
  > For any group of size $G=64$, the quantization step size is $\Delta = \frac{\max(W) - \min(W)}{2^B - 1}$, with rounding error bounded by $|\epsilon| \le \frac{\Delta}{2}$.
  >
  > When we scale column $c$ by $s_c > 1$, the scaled weight becomes $w' = w \cdot s_c$. During inference, multiplying by $s_c^{-1}$ yields:
  > $$\hat{w} = \frac{Q(w \cdot s_c)}{s_c} = w + \frac{\epsilon'}{s_c}$$
  > Because group size $G$ is small ($G \ll \text{total columns}$), scaling a single column does not significantly increase the group range $\Delta$. Therefore, **the effective quantization noise on salient columns drops by exactly $\frac{1}{s_c}$**!
  >
  > To find the optimal scale without overfitting, AWQ parameterizes $s = s_X^\alpha$ over a 1D exponent $\alpha \in [0, 1]$. By testing 20 discrete points on a calibration forward pass, the optimal $\alpha^*$ is found in seconds per layer via Frobenius norm minimization, without requiring gradient descent."
* **Transition:** *"Faculty often ask: why not use mixed precision, straight-through estimators, or low-rank decomposition?"*

---

### Slide 7: Architectural Design — Why Other Alternatives Fail (7:15 – 8:15)

* **Visual Anchor:** 3-Card Design Breakdown.
* **What to point at:** Walk through each card briefly.
* **What to say:**
  > "The authors systematically evaluated other potential architectures:
  >
  > 1. **Mixed Precision (1% FP16 + 99% INT4):** Keeping outlier channels in FP16 sounds intuitive, but irregular FP16 values require indexing metadata, cause warp divergence on GPUs, and break 128-bit SIMD registers on ARM NEON. In practice, mixed precision runs **slower** than pure FP16!
  > 2. **Straight-Through Estimator (STE) Gradient Descent:** Because the rounding operator $\lfloor \cdot \rceil$ has zero derivative almost everywhere, STE uses a surrogate gradient. On 70B models, this introduces high variance, unstable loss, and slow convergence.
  > 3. **Low-Rank / SVD Decomposition ($W = W_q + L_1 L_2^T$):** Isolating outliers into low-rank FP16 matrices requires two distinct GEMMs at runtime, increasing kernel launch overhead and memory bus contention.
  >
  > AWQ's coordinate transformation avoids all three pitfalls by maintaining a 100% uniform INT4 format."
* **Transition:** *"Now, let us examine our own implementation and empirical findings on Apple Silicon."*

---

### Slide 8: Our Implementation — Hardware Setup & Package Architecture (8:15 – 9:15)

* **Visual Anchor:** Left: Apple Silicon M1 specs & 3-stage workflow. Right: Table of `awq_lab` modules.
* **What to point at:** Emphasize that all code was built from scratch without third-party quantization wrappers.
* **What to say:**
  > "To verify the paper independently, we built an end-to-end Python package named **`awq_lab`** and evaluated it natively on Apple Silicon M1 using PyTorch with Metal Performance Shaders (`mps`).
  >
  > Our target model is **`SmolLM-135M`**, which uses the modern Llama architecture: RMSNorm, SwiGLU activations, and Rotary Positional Embeddings.
  >
  > As shown on the right, our architecture is clean and modular:
  > * `quantizer.py` handles group-wise affine quantization and 4-bit SIMD memory packing.
  > * `profiler.py` uses PyTorch forward hooks to capture per-channel activation statistics.
  > * `search.py` performs the 1D grid search over $\alpha \in [0, 1]$.
  > * `engine.py` applies quantization layer-by-layer across Attention and MLP blocks.
  > * `evaluator.py` benchmarks validation perplexity, logit cosine similarity, and KL divergence."
* **Transition:** *"Let us examine the exact quantitative metrics recorded from our experimental runs."*

---

### Slide 9: Experimental Findings & 9-Configuration Ablation Matrix (9:15 – 10:45)

* **Visual Anchor:** Top: 2 Empirical plots (`alpha_grid_search.pdf` & `logit_comparison.pdf`). Bottom: 9-Configuration Ablation Matrix Table.
* **What to point at:** Point to the convex bowl at $\alpha^* = 0.50$, then point to the $G=64$ row in the ablation table.
* **What to say:**
  > "Here are our empirical results on Apple M1:
  >
  > In the top-left plot, we observe a clear convex optimization bowl: at $\alpha = 0.0$ (Naive RTN), reconstruction MSE is $0.001536$. At $\alpha^* = 0.50$, MSE drops to $0.001107$—a **36.73% error reduction**! Pushing $\alpha > 0.6$ over-scales outliers and penalizes non-outliers, confirming the necessity of grid search.
  >
  > In the top-right plot, AWQ achieves **0.99908 logit cosine similarity** to the FP32 baseline and yields **5.3x lower KL divergence** than Naive RTN ($0.00198$ vs $0.01052$).
  >
  > Qualitatively, when prompted with *'Artificial neural networks learn by'*, Naive RTN degenerated into repeated list numbers (`### 1.1.1`), while AWQ generated a flawless scientific explanation of backpropagation.
  >
  > At the bottom, we conducted a full **9-configuration ablation study** sweeping bit-widths ($N \in \{4, 3, 2\}$) and group sizes ($G \in \{32, 64, 128\}$):
  > * **4-bit with $G=64$** represents the optimal tradeoff: 36.73% error reduction with only 6.25% metadata overhead.
  > * **3-bit with $G=32$** remains surprisingly viable (37.89% error reduction).
  > * **2-bit** exhibits a sharp $10\times$ MSE expansion, defining the theoretical breakdown boundary of uniform quantization."
* **Transition:** *"This brings us to our critical analysis of the paper's limitations and our proposals for a Delta Paper."*

---

### Slide 10: Critical Limitations & Roadmap for a Delta Paper (10:45 – 12:00)

* **Visual Anchor:** Left Red Box: Limitations. Right Green Box: 4 Delta Paper Proposals.
* **What to point at:** Emphasize how your 4 proposals directly solve the 4 limitations.
* **What to say:**
  > "While AWQ is an exceptional contribution, our empirical work identified four key limitations, for which we propose four concrete research extensions for a **Delta Paper**:
  >
  > 1. **Static Calibration vs Dynamic Gating:** AWQ fixes $\alpha^*$ offline from 128 calibration tokens. But outlier distributions shift between Python code and conversational text. **Proposal 1:** Design a lightweight runtime gating network that modulates $\alpha(t)$ dynamically from sequence prompt embeddings.
  > 2. **The Long-Context KV-Cache Bottleneck:** AWQ only compresses static model weights. For long context ($>8\text{k}$ tokens), the dynamic KV-cache exceeds the weight footprint. **Proposal 2:** Extend activation-aware coordinate scaling to Key-Value projection matrices to achieve stable W4A4 KV-caching.
  > 3. **The 2-Bit Breakdown:** As proven in our ablation study, uniform 2-bit quantization collapses. **Proposal 3:** Implement *AWQ-Sparse*—keeping just the top 0.5% outlier channels in native FP16 via dedicated SIMD paths on the Apple Neural Engine.
  > 4. **Hardware Kernels for Apple Silicon:** TinyChat wrote CUDA kernels for Nvidia GPUs. **Proposal 4:** Author native Metal Performance Shaders exploiting Apple Silicon's unified memory architecture for sub-millisecond mobile execution."
* **Transition:** *"To conclude our presentation..."*

---

### Slide 11: Summary & Key Takeaways (12:00 – 12:45)

* **Visual Anchor:** Synthesis Table comparing Paper vs Our Study vs Delta Extension.
* **What to say:**
  > "In summary:
  > * AWQ establishes that dynamic activation magnitude, not static weight norm, dictates quantization saliency.
  > * Invariant coordinate scaling protects outlier channels without retraining and executes with zero runtime overhead via offline RMSNorm fusion.
  > * Our empirical verification on Apple Silicon M1 confirms a **36.73% MSE reduction**, **5.3x lower KL divergence**, and robust generation across a 9-configuration ablation matrix.
  > * Finally, our proposed Delta Paper roadmap provides a concrete path forward toward dynamic $\alpha(t)$ scaling, KV-cache compression, and native Apple Silicon Metal kernels.
  >
  > Thank you for your time. I welcome your questions."

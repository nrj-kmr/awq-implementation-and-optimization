# AWQ Oral Defense & Viva-Voce Master Preparation Guide

## M.Tech AI Lab Examination — Faculty Technical Q&A & Board Derivations

This document is your **private, exhaustive defense handbook**. It contains the exact mathematical justifications, architectural arguments, and empirical proofs needed to answer faculty questions during your presentation review.

---

## 1. The Core 4 Defense Questions (Direct Paper Arguments)

### Question 1: Why did AWQ choose Weight-Only Quantization (W4A16) instead of Joint Weight-Activation Quantization (W8A8 / W4A4)?

- **Professors' Intent:** Testing whether you understand the hardware roofline model and the difference between server batching vs on-device single-token generation.
- **Your Bulletproof Answer:**
  1. **Arithmetic Intensity & Roofline Bottleneck:**
     $$\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Memory Access (Bytes)}}$$
     In on-device interactive decoding (batch size = 1), decoding generates one token at a time. The computational cost for a model with $P$ parameters is $2P \times 1\text{ FLOP}$. However, to compute this, **every single weight tensor in the model must be streamed from DRAM to SRAM/registers**:
     $$\text{Intensity}_{\text{FP16}} = \frac{2P}{P \times 2\text{ Bytes}} = 1.0\text{ FLOP/Byte}$$
     Hardware like Apple M1 or RTX 4090 has a compute-to-bandwidth roofline threshold $>100\text{ FLOPs/Byte}$. Therefore, on-device generation is **$100\%$ memory-bandwidth bound**. Compute cores spend $>90\%$ of cycle time stalled waiting for weight memory transfer.
  2. **Memory Traffic Breakdown (Weights vs Activations):**
     For batch size 1, a single token activation vector requires only $1 \times D$ floats (a few kilobytes). Meanwhile, the weight matrix is $D_{out} \times D_{in}$ (megabytes to gigabytes). Weight DRAM traffic is **$79\times \text{ to } 1700\times$ larger** than activation memory traffic (Figure 3c in the paper).
  3. **Preserving Model Quality:**
     Activations contain severe dynamic outliers across time and sequence. Compressing activations below 8-bit causes extreme perplexity degradation. Because activations consume less than $1\%$ of memory traffic at batch size 1, quantizing activations offers **near-zero speedup on edge devices** while introducing high risk of output corruption.
     Quantizing weights to 4-bit (W4A16) cuts memory traffic by $4\times$, directly accelerating generation speed by up to $3.5\times - 3.9\times$ with zero activation degradation.

---

### Question 2: Why is the grid search for $\alpha$ strictly bounded to $[0, 1]$? What happens if $\alpha > 1$?

- **Professors' Intent:** Testing if you understand the tradeoff between protecting salient channels and penalizing non-salient channels within the same quantization group.
- **Your Bulletproof Answer:**
  1. **The Boundary Definitions:**
     $$s = s_X^\alpha$$
     - When $\alpha = 0$: $s = s_X^0 = 1 \implies$ No scaling is applied; this reduces to standard Naive Round-To-Nearest (RTN).
     - When $\alpha = 1$: $s = s_X^1 \implies$ Full proportional scaling; the salient weights are scaled aggressively by their full activation ratio.
  2. **The Group Dynamic Range Explosion (Why $\alpha > 1$ fails):**
     In group quantization (e.g., $G = 64$ or $128$), all weights within a group share the same step size:
     $$\Delta = \frac{\max(W') - \min(W')}{2^B - 1}$$
     - If $\alpha$ is moderately scaled ($\alpha \in [0.2, 0.4]$), the salient channel $w \cdot s$ is scaled, but the group dynamic range $(\max - \min)$ remains bounded by other elements or only expands slightly. The relative error on the salient weight drops by $\approx 1/s$.
     - **If $\alpha > 1$:** The scale factor $s$ becomes excessively large ($s \gg 2$). The salient weight $w \cdot s$ now completely dominates and explodes the dynamic range of the entire group. This causes the step size $\Delta$ to expand dramatically for **all remaining 99% of non-salient weights** in that group.
     - The quantization error of the non-salient weights is amplified by $\frac{\Delta'}{\Delta} > 1$.
  3. **Empirical Evidence from Paper Table 2:**
     - For OPT-6.7B: At $s = 1$ (RTN), PPL is $23.54$. At $s = 2$ ($\alpha \approx 0.3$), PPL improves to optimal **$11.92$** (only $4.9\%$ of groups saw increased $\Delta$).
     - At $s = 4$ ($\alpha > 1$), $21.2\%$ of groups saw an expanded $\Delta$, and PPL worsened to **$18.04$**.
     - In our own Apple M1 experiment: $\alpha = 0.3$ gave optimal MSE ($0.001107$), while $\alpha = 1.0$ caused MSE to explode to $0.004597$!

---

### Question 3: Does dividing activations by $S$ ($\text{diag}(s)^{-1} X$) introduce runtime latency or memory overhead during inference?

- **Professors' Intent:** Checking if you understand operator fusion and compiler graphs in transformer models.
- **Your Bulletproof Answer:**
  - **Answer:** It introduces **exactly ZERO runtime operations, ZERO memory reads, and ZERO latency**.
  - **The Mathematical Proof:**
    In modern Transformer blocks (such as LLaMA, Mistral, SmolLM), all linear projections receive input directly from normalization layers (RMSNorm or LayerNorm):
    $$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$
    where $\gamma \in \mathbb{R}^d$ is the learnable gain parameter vector.
    In AWQ, the input to the linear layer is transformed by $\text{diag}(s)^{-1}$:
    $$X' = \text{RMSNorm}(x) \odot s^{-1} = \left( \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma \right) \odot s^{-1}$$
    By associativity of element-wise multiplication:
    $$X' = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \big(\gamma \odot s^{-1}\big)$$
    Therefore, we define a pre-fused gain vector:
    $$\gamma_{\text{fused}} = \gamma \odot s^{-1}$$
    This vector $\gamma_{\text{fused}}$ is computed **offline once** before deployment and saved directly into the model checkpoint.
  - During inference: The hardware executes standard RMSNorm using $\gamma_{\text{fused}}$. The scaling is applied simultaneously during normalization without a single extra instruction!

---

### Question 4: Why does AWQ outperform GPTQ on coding, math, and multimodal vision-language models?

- **Professors' Intent:** Testing whether you grasp the fundamental theoretical distinction between second-order parameter regression (GPTQ) versus coordinate transformation (AWQ).
- **Your Bulletproof Answer:**
  1. **GPTQ Modifies Weights to Fit Calibration Data:**
     GPTQ quantizes weights sequentially and compensates for the quantization error by updating all remaining unquantized weights using inverse Hessian regression:
     $$\Delta W_{:, j} = - \frac{w_q - w}{[H^{-1}]_{qq}} \cdot H^{-1}_{:, q}, \quad H = 2 X X^T$$
     Because it explicitly forces the unquantized weights to minimize error on the specific calibration sequences, **it overfits the calibration domain**. When the model encounters out-of-domain tasks (such as Python code syntax, multi-image visual tokens, or instruction prompts), the altered weights corrupt subtle representations.
  2. **AWQ Preserves Original Learned Parameters Intact:**
     AWQ does **NOT** alter the unquantized weights. It does not perform error compensation or regression. It merely performs coordinate rescaling ($W \cdot S$ and $S^{-1} X$), which preserves the original geometric feature relationships.
  3. **Calibration Data Robustness (Paper Figure 8 & Tables 6–8):**
     - AWQ achieves convergence using only **16 to 32 calibration sequences** ($10\times$ less data than GPTQ's 128–256 sequences).
     - On cross-domain evaluation (calibrating on PubMed and evaluating on Enron), AWQ’s perplexity changes by only $+0.5$, whereas GPTQ degrades by $+2.3 \text{ to } +4.9$.
     - On VILA-7B and VILA-13B across 11 visual benchmarks (Table 7), AWQ achieves **lossless zero-accuracy drop**, whereas GPTQ degrades on visual reasoning.

---

## 2. Advanced Technical Viva-Voce Questions

### Question 5: Can you derive the error reduction factor on the blackboard?

- **Derivation to write on the board:**
  1. Let a group of weights have range $[W_{\min}, W_{\max}]$. Step size for $B$-bit quantization:
     $$\Delta = \frac{W_{\max} - W_{\min}}{2^B - 1}$$
  2. Rounding a salient weight $w$ produces rounding error $\epsilon = Q(w) - w$, where $|\epsilon| \le \frac{\Delta}{2}$.
  3. Relative error on the unscaled salient weight:
     $$\text{RelErr}(w) = \frac{|\epsilon|}{|w|} \approx \frac{\Delta}{2|w|}$$
  4. Now multiply column $c$ by scalar $s > 1 \implies w' = w \cdot s$.
  5. The new step size is $\Delta'$. Since $G \ge 64$, the dynamic range of the group is determined by many weights, so $\Delta'$ increases only slightly ($\Delta' \ll s \Delta$).
  6. Quantizing the scaled weight gives $Q(w \cdot s) = w \cdot s + \epsilon'$, with $|\epsilon'| \le \frac{\Delta'}{2}$.
  7. During inference matrix multiplication, the de-scaling factor $s^{-1}$ is applied:
     $$\hat{w} = \frac{Q(w \cdot s)}{s} = \frac{w \cdot s + \epsilon'}{s} = w + \frac{\epsilon'}{s}$$
  8. The effective error on the original weight is now:
     $$\text{Effective Error} = \frac{\epsilon'}{s} \approx \frac{\Delta'}{2s}$$
     Since $\Delta' \approx \Delta$, the effective error has been reduced by a factor of $\mathbf{\frac{1}{s}}$!

---

### Question 6: Why did Naive Mixed-Precision (keeping 1% salient weights in FP16) run SLOWER on real hardware despite using less memory?

- **Faculty Insight:** Hardware systems knowledge.
- **Your Answer:**
  1. **Non-Contiguous Memory Layout:** Modern SIMD vector engines (ARM NEON 128-bit on Apple Silicon, AVX-512 on x86) and GPU Tensor Cores require regular, contiguous memory layouts where 4-bit elements are packed consecutively into 32-bit/64-bit words.
  2. **Branch Divergence & Indirection:** If 1% of the weights in a row are FP16 and 99% are INT4, the hardware cannot execute a vectorized instruction. It must either:
     - Store index pointers for where the FP16 weights are located (adding pointer chasing and cache misses).
     - Or introduce runtime conditional checks (`if (is_salient) load_fp16 else load_int4`), which causes **thread divergence** and stalls the execution pipeline.
  3. **Loss of Vectorization:** While the theoretical model size dropped, memory access throughput collapsed, resulting in higher execution latency than pure FP16.

---

### Question 7: How does AWQ differ from SmoothQuant?

- **SmoothQuant (Xiao et al., 2022):**
  - Target: **W8A8** (quantizes both weights and activations to 8 bits).
  - Purpose: Accelerate compute-bound large-batch inference on server GPUs (e.g. A100/H100) using INT8 Tensor Cores.
  - Mechanism: Migrates the quantization difficulty from activations to weights by dividing activations by $s$ and multiplying weights by $s$: $Y = (X \text{diag}(s)^{-1}) \cdot (\text{diag}(s) W)$.
  - Problem: Cannot push weights to 4-bit or 3-bit effectively because activations still have to be quantized to integers.
- **AWQ (Lin et al., 2024):**
  - Target: **W4A16** (quantizes weights to 4/3 bits; leaves activations in FP16).
  - Purpose: Accelerate memory-bandwidth bound batch=1 decoding on consumer edge devices (Apple Silicon, mobile, laptops).
  - Mechanism: Multiplies weights by $s$ to protect salient channels, and folds $s^{-1}$ into preceding LayerNorm. Activations remain in native FP16, completely bypassing activation quantization noise.

---

### Question 8: Why is Group-Wise Quantization ($G=64$ or $128$) needed instead of Per-Tensor or Per-Channel?

- **Per-Tensor Quantization:** A single scale $\Delta$ for the entire matrix ($D_{out} \times D_{in}$). A single extreme outlier anywhere in the matrix expands $\Delta$ for all millions of weights, destroying precision.
- **Per-Channel Quantization:** A separate scale $\Delta$ for each output channel (row). Better than per-tensor, but LLM weights exhibit localized dynamic range spikes within channels.
- **Group-Wise Quantization ($G=64/128$):** Divides each row into blocks of $G$ consecutive elements. Each group gets its own independent scale $\Delta$ and zero-point $Z$.
  - **Memory Overhead Tradeoff:** Storing one FP16 scale per 64 weights adds:
    $$\frac{16\text{ bits}}{64 \times 4\text{ bits}} = \frac{16}{256} = 0.0625 \implies \text{Only } 6.25\% \text{ storage overhead!}$$
  - In exchange for a negligible $6.25\%$ memory overhead, group quantization isolates outliers to small 64-element windows, preventing them from corrupting the rest of the channel.

---

### Question 9: What is the computational complexity of your calibration phase in Big-O notation?

- **Your Answer:**
  - Let $T$ be the number of calibration tokens (typically $T \approx 128$ to $512$ tokens), $D_{in}$ be input dimension, $D_{out}$ be output dimension, and $K$ be the number of grid search steps ($K = 20$).
  1. **Activation Profiling:** Single forward pass over $T$ tokens: $O(T \cdot D_{in} \cdot D_{out})$. Computing $s_X$ takes $O(T \cdot D_{in})$.
  2. **Grid Search Evaluation:** For each of the $K$ steps:
     - Coordinate scaling: $O(D_{out} \cdot D_{in})$
     - Group quantization: $O(D_{out} \cdot D_{in})$
     - Matrix multiplication reconstruction error $\| X W^T - X \hat{W}^T \|_F^2$: $O(T \cdot D_{in} \cdot D_{out})$
  3. **Total Layer Calibration Complexity:**
     $$O\Big( K \cdot T \cdot D_{in} \cdot D_{out} \Big)$$
  - Because $T$ is very small ($<512$) and $K=20$, this is linear with respect to parameter count.
  - Compare this to GPTQ, which requires inverting the Hessian matrix $H^{-1} = (2 X X^T)^{-1}$ with complexity $O(D_{in}^3)$ per layer! AWQ is significantly faster to calibrate and consumes orders of magnitude less memory during quantization.

---

### Question 10: How do you handle Residual Connections in your implementation?

- **Professors' Insight:** Verifying if you noticed architectural edge cases where LayerNorm fusion cannot be applied directly.
- **Your Answer:**
  - In a Transformer block:
    $$x_{l+1} = x_l + \text{SelfAttn}(x_l)$$
  - The inputs to $Q, K, V$ projections and FFN Gate/Up projections are preceded by RMSNorm, so $S^{-1}$ is fused directly into the RMSNorm weights.
  - However, for Attention Output Projection (`o_proj`) and FFN Down Projection (`down_proj`), their outputs are added directly back to the residual stream:
    $$\text{Residual} = x_{in} + W_{down} \cdot x_{intermediate}$$
  - We cannot scale the output of $W_{down}$ by an arbitrary vector because it would corrupt the residual addition.
  - Therefore, in AWQ, **channel scaling is applied strictly to the INPUT activations of linear layers**, not the outputs. For `down_proj`, $S^{-1}$ is folded into the preceding activation function or linear layer weights ($W_{gate}$ / $W_{up}$), keeping the residual stream unscaled and clean!

---

### Question 11: What if the calibration dataset is completely out-of-domain (e.g., calibrated on conversational English, tested on C++ code or medical diagnosis)?

- **Your Answer:**
  - The authors tested this exact failure mode in Figure 8b by calibrating on PubMed (biomedical) and evaluating on Enron (corporate emails), and vice versa.
  - **Result:** AWQ's perplexity only degraded by $+0.5$ to $+0.6$, whereas GPTQ degraded by $+2.3$ to $+4.9$.
  - **Why AWQ is robust to domain mismatch:** Activation outliers in LLMs are primarily **structural features** tied to the transformer's internal coordinate geometry (such as punctuation delimiters, position indicators, and residual magnitude anchors) rather than specific vocabulary words. The top $1\%$ outlier channels remain consistent across diverse text domains.

---

## 3. Summary of Concrete Numbers to Quote During Review

Keep these numbers in mind if asked about your experimental findings on the Apple Silicon M1:

| Finding | Metric / Value |
| :--- | :--- |
| **Model Evaluated** | `HuggingFaceTB/SmolLM-135M` (135 Million parameters, Llama architecture) |
| **Quantization Scheme** | Uniform Affine INT4 with Group Size $G=64$ |
| **Layer 0 Activation Outlier Ratio** | **$17.26\times$** (Peak channel $0.8995$ vs mean $0.0521$) |
| **Saliency Overlap ($\vert W \vert$ vs $s_X$)** | **Only 50%** (Top-10 overlap is 5/10; $\vert W \vert$ missed channel 303 completely) |
| **Reconstruction Error Reduction** | **$36.73\%$ Average MSE Reduction** over Naive RTN across quantized blocks |
| **Optimal Grid Search Alpha** | $\mathbf{\alpha^* \approx 0.25\text{--}0.30}$ |
| **Logit Cosine Similarity to FP32** | RTN: $0.99231 \implies$ **AWQ: $0.99908$** |
| **Logit KL Divergence to FP32** | RTN: $0.01052 \implies$ **AWQ: $0.00198$** (**$5.3\times$ lower probability shift!**) |
| **Generation Comparison** | RTN degenerated into number listing (`### 1.1.1`), while AWQ produced correct scientific continuation (`backpropagation`). |

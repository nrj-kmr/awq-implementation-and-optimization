# AWQ: Activation-aware Weight Quantization on Apple Silicon

## M.Tech AI Lab Project & Academic Seminar Submission

This repository contains the complete implementation, empirical verification, academic presentation, and faculty defense materials for the paper:
> **"AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration"**
> *Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, Song Han (MIT HAN Lab)*
> Published at **MLSys 2024 / NeurIPS 2023** (arXiv:2306.00978).

---

## 1. Project Deliverables

| Deliverable | Description | File Path |
| :--- | :--- | :--- |
| 📜 **IEEE Research Paper (PDF)** | 6-page publication-ready IEEE conference research paper with active hyperlinks | [`paper/ieee_awq_paper.pdf`](paper/ieee_awq_paper.pdf) |
| 📝 **Paper LaTeX Source** | Agnostic IEEE conference source with derivations and 9-configuration ablation matrix | [`paper/ieee_awq_paper.tex`](paper/ieee_awq_paper.tex) |
| 📄 **Academic Presentation (PDF)** | 12-slide minimalist Beamer presentation with embedded vector figures | [`presentation/lab_presentation.pdf`](presentation/lab_presentation.pdf) |
| 📝 **Presentation LaTeX Source** | Clean LaTeX Beamer source code (`dove` academic theme, 0 warnings) | [`presentation/lab_presentation.tex`](presentation/lab_presentation.tex) |
| 🚀 **Live Demo Runner** | Single-command 15-second benchmark demonstration for oral evaluation | [`run_demo.sh`](run_demo.sh) |
| 🗣️ **Oral Defense & Speaker Notes** | Timed (12-14 min) word-for-word presentation script with visual focal points | [`presentation/notes/SPEAKER_NOTES_SCRIPT.md`](presentation/notes/SPEAKER_NOTES_SCRIPT.md) |
| 🛡️ **Faculty Defense Q&A Guide** | 11 comprehensive viva-voce technical questions, proofs, and board derivations | [`presentation/notes/FACULTY_DEFENSE_QA.md`](presentation/notes/FACULTY_DEFENSE_QA.md) |
| 💡 **Delta Proposal Roadmap** | Architectural delta extensions (dynamic gating, KV-cache, sparse outliers) | [`presentation/notes/DELTA_PAPER_PROPOSAL.md`](presentation/notes/DELTA_PAPER_PROPOSAL.md) |
| 📊 **High-Resolution Figures** | Vector graphics generated from Apple Silicon M1 profiling | [`figures/`](figures/) |
| 📦 **`awq_lab` Python Package** | Clean, modular PyTorch quantization engine with forward hooks & grid search | [`awq_lab/`](awq_lab/) |
| 🧪 **Unit Test Suite** | Automated test suite validating quantization error bounds, packing, & hooks | [`tests/test_awq.py`](tests/test_awq.py) |
| 🛠️ **Helper Figure Generator** | Automated publication vector figure generator | [`helpers/generate_presentation_figures.py`](helpers/generate_presentation_figures.py) |
| ⚙️ **Project Automation** | Automated pipeline for build, test, figure generation, and cleaning | [`Makefile`](Makefile) |
| 📈 **Ablation Matrix (JSON)** | Full 9-configuration sweep ($N \in \{4,3,2\} \times G \in \{32,64,128\}$) | [`results/ablation_results.json`](results/ablation_results.json) |
| 📈 **Empirical Metrics (JSON)** | Exact numerical results recorded from Apple Silicon M1 run | [`results/results_summary.json`](results/results_summary.json) |
| 📑 **Original Paper** | Official NeurIPS/MLSys published paper PDF | [`base_paper/2306.00978v6.pdf`](base_paper/2306.00978v6.pdf) |

---

## 2. Environment Setup & Getting Started

This project is built to run cross-platform with first-class hardware acceleration:

* **Apple Silicon (M1/M2/M3/M4):** Accelerated via PyTorch Metal Performance Shaders (`mps`).
* **NVIDIA GPU:** Accelerated via CUDA.
* **CPU:** Portable fallback across Linux, macOS, and Windows.

### Prerequisites

* Python **3.10** or **3.11**
* `git`
* (Optional) LaTeX distribution (e.g., MacTeX, TeX Live, BasicTeX) to compile papers and presentation slides.

---

### Step 1: Clone the Repository

```bash
git clone https://github.com/nrj-kmr/awq-implementation-and-optimization.git
cd awq-implementation-and-optimization
```

---

### Step 2: Create an Isolated Virtual Environment

Choose either **Conda / Mamba** (recommended for Apple Silicon & scientific computing) or standard Python **venv**:

#### Method A: Using Conda or Miniforge / Mamba (Recommended)

You can create the environment directly from the provided declarative `environment.yml`:

```bash
# Option 1: Create directly from environment.yml
conda env create -f environment.yml
conda activate awq

# Option 2: Or create a fresh environment manually
conda create -n awq python=3.10 -y
conda activate awq
```

#### Method B: Using Standard Python `venv` (Zero external tools)

```bash
python3 -m venv .venv

# On macOS / Linux:
source .venv/bin/activate

# On Windows:
# .venv\Scripts\activate
```

---

### Step 3: Install Dependencies & Package

Install the required dependencies and install the project in **editable mode**:

```bash
# 1. Install dependencies from requirements.txt
pip install -r requirements.txt

# 2. Install the awq-lab package in editable mode
pip install -e .
# Alternatively: make install
```

> [!NOTE]
> Installing with `pip install -e .` links the `awq_lab` Python package into your site-packages and installs the CLI entry points (`awq-run`, `awq-ablate`, `awq-figures`), allowing you to invoke them from any directory.

---

### Step 4: Verify Installation (Sanity Check)

Run the automated unit test suite to verify that PyTorch tensor operations, group quantization, coordinate scaling, and hooks are functioning correctly:

```bash
make test
# Or directly:
python -m unittest discover tests
```

*Expected output: `Ran 5 tests in ~0.6s ... OK`.*

---

### Step 5: Run Your First Benchmark

Run the live AWQ benchmark on `HuggingFaceTB/SmolLM-135M`:

```bash
# Option 1: One-command demo script (runs in ~15 seconds)
./run_demo.sh

# Option 2: Via Makefile target
make run

# Option 3: Via CLI directly with custom parameters
awq-run --bits 4 --group_size 64 --num_layers 2
```

---

## 3. Workflow & Automation Commands

A unified [`Makefile`](Makefile) automates all project tasks:

```bash
# View all available targets and descriptions
make help

# Run unit test suite (5 test cases)
make test

# Run live AWQ benchmark (SmolLM-135M, W4A16, G=64)
make run

# Run complete 9-configuration grid ablation (N in {4,3,2} x G in {32,64,128})
make ablations

# Re-generate all publication vector graphics in figures/
make figures

# Compile the IEEE conference paper (paper/ieee_awq_paper.pdf)
make paper

# Compile the academic Beamer presentation (presentation/lab_presentation.pdf)
make presentation

# Compile both IEEE paper and presentation slides
make all

# Clean up __pycache__, bytecode, build artifacts, and LaTeX temp files
make clean
```

### Direct CLI Commands

After `pip install -e .`, the following console commands are available anywhere in your active environment:

```bash
# Run quantization benchmark with custom bit-width, group size, and layers
awq-run --bits 4 --group_size 64 --num_layers 6

# Run full ablation grid search
awq-ablate

# Generate presentation & paper figures
awq-figures
```

### Keeping the Workspace Clean

To prevent Python from generating `__pycache__` bytecode directories during development, you can set:

```bash
export PYTHONDONTWRITEBYTECODE=1
```

Or run `make clean` at any time to purge intermediate cache and LaTeX build files while preserving your compiled PDFs.

---

## 4. Repository Architecture

The project is cleanly modularized into dedicated directories:

```text
AWQ-paper/
├── pyproject.toml            # PEP 517/518/621 packaging & CLI entry points
├── environment.yml           # Conda/Mamba declarative environment specification
├── requirements.txt          # Pip dependencies specification
├── Makefile                  # Developer workflow automation
├── .gitignore                # Comprehensive bytecode, secrets, and LaTeX filter
├── run_demo.sh               # Executable 15-second benchmark runner
├── main_experiment.py        # Benchmark driver script (outputs to results/)
├── run_ablations.py          # Grid ablation driver script (outputs to results/)
├── test_awq_on_mac.py        # Standalone prototype verification script
│
├── awq_lab/                  # Core importable Python package
│   ├── __init__.py           # Package interface & symbol exports
│   ├── quantizer.py          # Group-wise uniform quantization (RTN) & coordinate scaling
│   ├── profiler.py           # PyTorch forward hooks for activation capture & outlier profiling
│   ├── search.py             # 1D grid search over alpha in [0.0, 1.0] minimizing Frobenius error
│   ├── engine.py             # Multi-layer Transformer quantizer (Attention & MLP)
│   └── evaluator.py          # Greedy generation, validation perplexity, and logit KL divergence
│
├── paper/                    # IEEE Conference Research Paper
│   ├── IEEEtran.cls          # Official IEEEtran class file
│   ├── ieee_awq_paper.tex    # Pure agnostic IEEE conference paper LaTeX source
│   └── ieee_awq_paper.pdf    # Compiled 6-page publication-ready PDF with active hyperlinks
│
├── helpers/                  # Utilities & figure generation
│   ├── __init__.py           # Helpers package marker
│   └── generate_presentation_figures.py # Matplotlib vector figure generator
│
├── presentation/             # Academic presentation deck & defense documentation
│   ├── lab_presentation.tex  # 12-slide academic Beamer deck (widescreen 16:9)
│   ├── lab_presentation.pdf  # Compiled presentation deck (zero visual overlaps)
│   └── notes/                # Seminar notes & viva defense materials
│       ├── FACULTY_DEFENSE_QA.md     # 11 rigorous technical viva questions and board derivations
│       ├── SPEAKER_NOTES_SCRIPT.md   # Word-for-word 12-14 minute oral presentation script
│       └── DELTA_PAPER_PROPOSAL.md   # Architectural delta proposals (dynamic alpha, KV cache)
│
├── results/                  # Empirical metrics & experimental summaries
│   ├── results_summary.json  # Recorded metrics from Apple Silicon M1 run
│   └── ablation_results.json # Full 9-configuration sweep (bits x group_size)
│
├── figures/                  # Vector graphics for publications & slides
│   ├── roofline_memory_wall.pdf
│   ├── activation_outliers.pdf
│   ├── alpha_grid_search.pdf
│   ├── logit_comparison.pdf
│   └── awq_scaling_pipeline.pdf
│
├── tests/                    # Unit test suite
│   └── test_awq.py           # Automated unit tests (5 test cases)
│
└── base_paper/               # Reference literature
    └── 2306.00978v6.pdf      # Original AWQ paper (MLSys 2024 / NeurIPS 2023)
```

---

## 5. Empirical Findings on Apple Silicon M1

Tested on `HuggingFaceTB/SmolLM-135M` (Llama architecture) using PyTorch 2.3.0 with Metal Performance Shaders (`device = 'mps'`):

| Metric | Baseline FP32 | Naive RTN (INT4-g64) | AWQ (INT4-g64) | Impact / Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Layer Reconstruction MSE** | 0.0000 | 0.001536 | **0.001107** | **$36.73\%$ Average Error Reduction** |
| **Validation Perplexity (PPL)** | 4.77 | 4.88 | **4.87** | Superior output stability |
| **Logit Cosine Similarity** ($\uparrow$) | 1.0000 | 0.99231 | **0.99908** | Near-lossless directional alignment |
| **Logit KL Divergence** ($\downarrow$) | 0.0000 | 0.01052 | **0.00198** | **$5.3\times$ Lower Distribution Shift** |
| **Activation Outlier Ratio** | — | — | **$17.26\times$** | Channel 303: $0.8995$ vs mean $0.0521$ |
| **Saliency Overlap ($\vert W \vert$ vs $s_X$)** | — | — | **50.0%** | Weight norm misses $5/10$ true outliers |

### Qualitative Output Verification

* **Prompt:** *"Artificial neural networks learn by"*
  * **FP32:** *"adjusting the weights of the connections between neurons..."*
  * **Naive RTN (Degraded):** *"adjusting weights and biases to minimize error... ### 1.1.1"* *(loss of attention focus; degenerate numbering)*.
  * **AWQ (Coherent):** *"adjusting the weights of the connections between neurons. This process is called backpropagation."* *(flawless scientific reasoning preserved)*.

---

## 6. Summary of Mathematical Formulation

AWQ performs an exact coordinate transformation before quantization:
$$Y = W X = \big(W \cdot \text{diag}(s)\big) \cdot \big(\text{diag}(s)^{-1} X\big) = W' X'$$

Where the per-channel protective scale $s$ is parameterized along a 1D grid space governed by activation magnitude $s_X = \mathbb{E}[|X|]$:
$$s = \left(\frac{s_X}{\text{mean}(s_X)}\right)^\alpha, \quad \alpha^* = \arg\min_{\alpha \in [0, 1]} \left\| Q(W \cdot \text{diag}(s)) \cdot (\text{diag}(s)^{-1} X) - W X \right\|_F^2$$

The inverse scaling matrix $S^{-1} = \text{diag}(s)^{-1}$ is folded offline into preceding layer normalization gains (RMSNorm):
$$\gamma_{\text{fused}} = \gamma \odot s^{-1} \implies \textbf{Zero runtime FLOPs or latency overhead.}$$

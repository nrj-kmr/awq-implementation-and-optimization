#!/usr/bin/env bash
# ==============================================================================
# AWQ Empirical Live Demonstration Runner
# Prepared for M.Tech AI Lab Project Evaluation (IIIT Vadodara)
# Author: Neeraj Kumar
# ==============================================================================

set -e

# Detect Python interpreter (priority: miniforge awq env, then current python)
if [ -f "/opt/homebrew/Caskroom/miniforge/base/envs/awq/bin/python" ]; then
    PYTHON_EXEC="/opt/homebrew/Caskroom/miniforge/base/envs/awq/bin/python"
elif command -v python3 &> /dev/null; then
    PYTHON_EXEC="$(which python3)"
else
    echo "Error: Python 3 interpreter not found!"
    exit 1
fi

echo "========================================================================"
echo " AWQ: ACTIVATION-AWARE WEIGHT QUANTIZATION (LIVE DEMO)"
echo " Hardware: Apple Silicon M1 (ARM64 Unified Memory Architecture)"
echo " Python  : $PYTHON_EXEC"
echo " Date    : $(date)"
echo "========================================================================"
echo ""
echo "Select evaluation mode to execute:"
echo "  1) Quick Live Demo       (Main benchmark: W4G64, 2 layers, ~15s) [DEFAULT]"
echo "  2) Full Ablation Matrix  (9 configurations: N in {4,3,2} x G in {32,64,128}, ~6s)"
echo "  3) Re-generate Figures   (Publication vector plots in figures/, ~8s)"
echo "  4) Compile IEEE Paper    (Compiles paper/ieee_awq_paper.pdf)"
echo "  5) Compile Presentation  (Compiles presentation/lab_presentation.pdf)"
echo "  6) Run Everything        (Demo + Ablation + Paper + Presentation)"
echo ""
read -t 10 -p "Enter choice [1-6, default=1 in 10s]: " choice || choice=1
echo ""

case "${choice:-1}" in
    1)
        echo "--> [1/1] Running AWQ Empirical Benchmark (W4A16, G=64, 2 layers)..."
        "$PYTHON_EXEC" main_experiment.py --bits 4 --group_size 64 --num_layers 2
        ;;
    2)
        echo "--> [1/1] Running 9-Configuration Ablation Sweep..."
        "$PYTHON_EXEC" run_ablations.py
        ;;
    3)
        echo "--> [1/1] Regenerating Academic Publication Figures..."
        "$PYTHON_EXEC" helpers/generate_presentation_figures.py
        ;;
    4)
        echo "--> [1/1] Compiling IEEE Research Paper..."
        cd paper && pdflatex -interaction=nonstopmode ieee_awq_paper.tex && cd ..
        echo "--> Done: paper/ieee_awq_paper.pdf"
        ;;
    5)
        echo "--> [1/1] Compiling Beamer Presentation..."
        cd presentation && pdflatex -interaction=nonstopmode lab_presentation.tex && cd ..
        echo "--> Done: presentation/lab_presentation.pdf"
        ;;
    6)
        echo "--> Running Complete Benchmark and Compilation Pipeline..."
        "$PYTHON_EXEC" main_experiment.py --bits 4 --group_size 64 --num_layers 2
        "$PYTHON_EXEC" run_ablations.py
        "$PYTHON_EXEC" helpers/generate_presentation_figures.py
        cd paper && pdflatex -interaction=nonstopmode ieee_awq_paper.tex && cd ..
        cd presentation && pdflatex -interaction=nonstopmode lab_presentation.tex && cd ..
        echo "--> All artifacts compiled and verified successfully!"
        ;;
    *)
        echo "Invalid selection. Running Default Live Demo..."
        "$PYTHON_EXEC" main_experiment.py --bits 4 --group_size 64 --num_layers 2
        ;;
esac

echo ""
echo "========================================================================"
echo " Live Execution Completed Successfully!"
echo "========================================================================"

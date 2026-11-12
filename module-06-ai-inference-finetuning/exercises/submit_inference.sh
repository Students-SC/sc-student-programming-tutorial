#!/bin/bash
# ---------------------------------------------------------------------------
# submit_inference.sh -- Run LLM inference on a GPU node
#
# Submit:  sbatch submit_inference.sh [script.py]
# ---------------------------------------------------------------------------
#SBATCH --job-name=inference
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err

set -euo pipefail

echo "LLM Inference -- Job ${SLURM_JOB_ID:-(not in Slurm)} on $(hostname)"
echo ""

# The shared, read-only Python environment that the instructors built
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
VENV_DIR="$SC26_SHARED_DIR/sc26_venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: shared Python environment not found at $VENV_DIR" >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Load models from the shared, read-only Hugging Face cache, without
# contacting the Hub (see setup/shared-env/README.md).
export HF_HUB_CACHE="$SC26_SHARED_DIR/sc26_hf_cache/hub"
export HF_HUB_OFFLINE=1

echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "GPU available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# The Python script to run is the first argument (default: the example):
#   sbatch submit_inference.sh                    # Exercise 2
#   sbatch submit_inference.sh run_inference.py   # Exercise 3
SCRIPT="${1:-../examples/simple_inference.py}"
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found: $SCRIPT" >&2
    exit 1
fi
echo "Running: $SCRIPT"
echo ""
python3 "$SCRIPT"

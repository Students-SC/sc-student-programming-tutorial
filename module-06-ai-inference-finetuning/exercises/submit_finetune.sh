#!/bin/bash
# ---------------------------------------------------------------------------
# submit_finetune.sh -- Run LoRA fine-tuning on a GPU node
#
# Submit:  sbatch submit_finetune.sh [script.py]
# ---------------------------------------------------------------------------
#SBATCH --job-name=finetune
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

echo "LoRA Fine-Tuning -- Job $SLURM_JOB_ID on $(hostname)"
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

echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# The Python script to run is the first argument (default: the exercise)
SCRIPT="${1:-finetune_lora.py}"
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found: $SCRIPT" >&2
    exit 1
fi
echo "Running: $SCRIPT"
echo ""
python3 "$SCRIPT"

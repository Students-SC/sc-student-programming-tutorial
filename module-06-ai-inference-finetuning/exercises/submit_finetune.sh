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

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set. Ask an instructor for help." >&2
    exit 1
fi
VENV_DIR="$WORK/sc26_venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: Python venv not found at $VENV_DIR" >&2
    echo "       Create it from the repo root with: sbatch setup/setup_venv.sh" >&2
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

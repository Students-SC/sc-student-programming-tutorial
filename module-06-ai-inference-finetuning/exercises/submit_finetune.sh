#!/bin/bash
# ---------------------------------------------------------------------------
# submit_finetune.sh -- Run LoRA fine-tuning on a GPU node
#
# Submit:  sbatch submit_finetune.sh
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
source "$VENV_DIR/bin/activate"

echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

echo "--- GPU memory before ---"
rocm-smi --showmeminfo vram 2>/dev/null | head -10
echo ""

python3 finetune_lora.py

echo ""
echo "--- GPU memory after ---"
rocm-smi --showmeminfo vram 2>/dev/null | head -10

#!/bin/bash
# ---------------------------------------------------------------------------
# submit_inference.sh -- Run LLM inference on a GPU node
#
# Submit:  sbatch submit_inference.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=inference
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=inference_%j.out
#SBATCH --error=inference_%j.err

echo "LLM Inference -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set. Ask an instructor for help." >&2
    exit 1
fi
VENV_DIR="$WORK/sc26_venv"
source "$VENV_DIR/bin/activate"

echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "ROCm:    $(python3 -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run the inference script (change to run_inference.py for the exercise)
python3 ../examples/simple_inference.py

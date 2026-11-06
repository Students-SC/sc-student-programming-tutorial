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

echo "LLM Inference -- Job $SLURM_JOB_ID on $(hostname)"
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
echo "ROCm:    $(python3 -c 'import torch; print(torch.cuda.is_available())')"
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

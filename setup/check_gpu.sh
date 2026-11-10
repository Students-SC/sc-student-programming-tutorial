#!/bin/bash
# ---------------------------------------------------------------------------
# check_gpu.sh -- Check that your jobs can use a GPU and the tutorial's
# shared Python environment
#
# Submit (from the repo root):  sbatch setup/check_gpu.sh
# Module 2 explains what each line of this script does.
# ---------------------------------------------------------------------------
#SBATCH --job-name=check-gpu
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00
#SBATCH --output=check_gpu_%j.out
#SBATCH --error=check_gpu_%j.err

set -euo pipefail

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
VENV_DIR="$SC26_SHARED_DIR/sc26_venv"

echo "GPU check -- job ${SLURM_JOB_ID:-none} on $(hostname)"
echo ""

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "Status:  FAILED -- shared Python environment not found at $VENV_DIR"
    echo "         Please ask an instructor for help."
    exit 1
fi
source "$VENV_DIR/bin/activate"

python3 - <<'EOF'
import sys
import torch

print(f"Python env: {sys.prefix}")
print(f"PyTorch:    {torch.__version__}")
if not torch.cuda.is_available():
    print("Status:     FAILED -- PyTorch cannot see a GPU. Please ask an instructor for help.")
    raise SystemExit(1)

x = torch.rand(4096, 4096, device="cuda")
(x @ x).sum().item()  # a small matrix multiply on the GPU
print(f"GPU:        {torch.cuda.get_device_name(0)}")
print("Status:     PASSED")
EOF

#!/bin/bash
# ---------------------------------------------------------------------------
# check_gpu.sh -- Check that your jobs can use a GPU and the tutorial's
# shared Python environment
#
# Submit (from the repo root):  sbatch setup/check_gpu.sh
# Module 2 explains what each line of this script does.
# ---------------------------------------------------------------------------
#SBATCH --job-name=check-gpu
#SBATCH -A TRN001 --partition=batch  --reservation=sc26studentprogram
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00
#SBATCH --output=check_gpu_%j.out
#SBATCH --error=check_gpu_%j.err

set -euo pipefail

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/lustre/orion/stf007/world-shared/subil/SC26_student_program}"
VENV_DIR="$SC26_SHARED_DIR/sc26_venv"
module load miniforge3
module load rocm/7.2

echo "GPU check -- job ${SLURM_JOB_ID:-none} on $(hostname)"
echo ""

conda activate $VENV_DIR

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

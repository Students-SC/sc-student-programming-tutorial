#!/bin/bash
# ---------------------------------------------------------------------------
# setup_venv.sh -- Create a Python virtual environment for the AI modules
#
# This script is meant to be run as a Slurm batch job on a compute node:
#
#   sbatch --partition=mi2101x --time=10:00 --ntasks=1 setup/setup_venv.sh
#
# It creates a venv under /work1 (which has much more storage than $HOME)
# with PyTorch (ROCm), transformers, and related packages for Modules 6 and 7.
# ---------------------------------------------------------------------------
#SBATCH --job-name=setup-venv
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=setup_venv_%j.out

set -euo pipefail

# Place the venv on /work1 (large shared storage) rather than $HOME (small quota).
# $WORK is set by the cluster environment to /work1/<project-id>/$USER.
if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set." >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi
VENV_DIR="$WORK/sc26_venv"

echo "=== Setting up Python virtual environment ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Target:  $VENV_DIR"
echo ""

if [ -d "$VENV_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment with Python 3.12..."
python3.12 -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing PyTorch with ROCm 7.2 support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

echo ""
echo "Installing AI/ML libraries..."
pip install transformers datasets peft accelerate bitsandbytes
pip install openai requests

echo ""
echo "Installing aider (CLI coding agent for Module 7)..."
pip install aider-chat

echo ""
echo "=== Verifying installation ==="
python3 -c "
import torch
print(f'PyTorch version:  {torch.__version__}')
print(f'ROCm available:   {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device:       {torch.cuda.get_device_name(0)}')
    print(f'GPU memory:       {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=== Setup complete ==="
echo "To activate:  source $VENV_DIR/bin/activate"

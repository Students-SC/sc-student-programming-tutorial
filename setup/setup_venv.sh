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

module load miniforge3
module load rocm/7.2
if [[ $(conda env list | grep sc26_venv) ]]; then
    echo "Removing existing aider venv..."
    conda env remove -n sc26_aider_venv
fi

echo "Creating virtual environment with Python 3.12..."
# we'll create a conda virtual environment that is stored in the default location
# instead of specifying a location
# Python 3.12 is required since that is what the Aider release we are using works with
conda create -n sc26_venv python=3.12 -y

conda activate sc26_venv

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2

echo ""
echo "Installing AI/ML libraries..."
pip install transformers datasets peft accelerate bitsandbytes
pip install openai requests

# NOTE: aider is intentionally NOT installed here. It is a per-user client tool
# with dependencies (openai/pydantic versions) that conflict with this ML stack,
# so it lives in its own venv. Install it separately:  bash setup/install_aider.sh

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

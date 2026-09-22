#!/bin/bash
# ---------------------------------------------------------------------------
# install_aider.sh -- Install the aider CLI coding agent in its own venv
#
# aider is a per-user *client* (it just makes HTTP calls to the shared vLLM
# server). It lives in its OWN virtual environment, separate from the ML venv
# created by setup_venv.sh, because aider pins older versions of libraries like
# openai/pydantic that conflict with the PyTorch/transformers stack. Keeping
# them apart means upgrading one can never break the other.
#
# This is a pure-Python pip install with no GPU or compile step, so -- unlike
# setup_venv.sh -- it is fine to run directly on the login node (no Slurm job).
#
#   bash setup/install_aider.sh
#
# Afterwards, launch the agent with:  bash setup/launch_aider.sh
# ---------------------------------------------------------------------------

set -euo pipefail
module load miniforge3



# Pin the version so every developer (and, later, every student) runs the
# exact same agent. Bump this deliberately when you want to upgrade.
AIDER_VERSION="0.86.2"

echo "=== Installing aider (SC26 coding agent) ==="
echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "Version: aider-chat==$AIDER_VERSION"
echo ""

if [[ $(conda env list | grep sc26) ]]; then
    echo "Removing existing aider venv..."
    conda env remove -n sc26_aider_venv
fi
#
# we'll create a conda virtual environment that is stored in the default location
# instead of specifying a location
# Python 3.12 is required since that is what the Aider release we are using works with
conda create -n sc26_aider_venv python=3.12 -y

conda activate sc26_aider_venv


echo ""
echo "Installing aider-chat==$AIDER_VERSION ..."
pip install "aider-chat==$AIDER_VERSION"

echo ""
echo "=== Verifying installation ==="
aider --version

echo ""
echo "=== Setup complete ==="
echo "Launch the agent with:  bash setup/launch_aider.sh"

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

# The aider venv is per-user and lives under $WORK (large quota, per-user),
# alongside -- but separate from -- the ML venv ($WORK/sc26_venv).
if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set." >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi

AIDER_VENV="$WORK/sc26_aider_venv"

# Pin the version so every developer (and, later, every student) runs the
# exact same agent. Bump this deliberately when you want to upgrade.
AIDER_VERSION="0.86.2"

echo "=== Installing aider (SC26 coding agent) ==="
echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "Target:  $AIDER_VENV"
echo "Version: aider-chat==$AIDER_VERSION"
echo ""

if [[ -d "$AIDER_VENV" ]]; then
    echo "Removing existing aider venv..."
    rm -rf "$AIDER_VENV"
fi

echo "Creating virtual environment with Python 3.12..."
python3.12 -m venv "$AIDER_VENV"

# Use the venv's own pip/binaries directly (no activation needed).
"$AIDER_VENV/bin/python" -m pip install --upgrade pip

echo ""
echo "Installing aider-chat==$AIDER_VERSION ..."
"$AIDER_VENV/bin/python" -m pip install "aider-chat==$AIDER_VERSION"

echo ""
echo "=== Verifying installation ==="
"$AIDER_VENV/bin/aider" --version

echo ""
echo "=== Setup complete ==="
echo "Launch the agent with:  bash setup/launch_aider.sh"

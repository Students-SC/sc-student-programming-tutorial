#!/bin/bash
# ---------------------------------------------------------------------------
# launch_aider.sh -- Run aider configured for the SC26 tutorial vLLM server
#
# Aider is a CLI coding agent. This wrapper points it at the locally-served
# Qwen3-Coder model from Module 7's vLLM backend, so no API keys or external
# accounts are needed. It runs aider from its dedicated venv
# ($WORK/sc26_aider_venv, created by setup/install_aider.sh).
#
# By default aider runs in ASK mode: it reads the files you give it and explains
# what to do, but does not edit them -- understanding is the participant's job.
#
# Usage:
#     cd <project directory you want to ask about>
#     bash <repo>/setup/launch_aider.sh [aider args...]
#
# Examples:
#     bash setup/launch_aider.sh                          # ask-mode guidance
#     bash setup/launch_aider.sh --read README.md foo.c   # pin files as context
#
# Type /help inside aider to see commands, /exit to quit.
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Pre-flight checks ------------------------------------------------------
# aider's venv is per-user (under $WORK), but the agent server URL file is
# shared (under $SC26_SHARED_DIR) so all students hit the same vLLM endpoint.
if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set." >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/sc26dev/shared}"
URL_FILE="$SC26_SHARED_DIR/sc26_agent_server_url"
if [[ ! -f "$URL_FILE" ]]; then
    echo "ERROR: vLLM server URL file not found at $URL_FILE" >&2
    echo "       The agent backend isn't running. Ask an instructor for help." >&2
    exit 1
fi

# aider lives in its OWN per-user venv (separate from the ML venv sc26_venv),
# created by setup/install_aider.sh. We invoke its binary by absolute path so
# this never accidentally picks up some other aider on $PATH.
AIDER_VENV="$WORK/sc26_aider_venv"
AIDER_BIN="$AIDER_VENV/bin/aider"
if [[ ! -x "$AIDER_BIN" ]]; then
    echo "ERROR: aider not found at $AIDER_BIN" >&2
    echo "       Install it first (from the repo root): bash setup/install_aider.sh" >&2
    exit 1
fi

# --- Configure aider for the local vLLM server -----------------------------
export OPENAI_API_BASE="$(cat "$URL_FILE")"
export OPENAI_API_KEY="sk-no-key-needed"   # vLLM ignores this; aider needs *something*

MODEL="openai/Qwen/Qwen3-Coder-30B-A3B-Instruct"

echo "=== Launching aider (SC26) ==="
echo "  API:    $OPENAI_API_BASE"
echo "  Model:  $MODEL"
echo "  CWD:    $(pwd)"
echo ""
echo "Type /help inside aider for commands, /exit to quit."
echo ""

# Default to ask mode: aider reasons about the participant's actual files and
# explains what to do, but never edits them -- understanding is the
# participant's job. Module 7's capstone deliberately overrides this with
# `--chat-mode code` to introduce the full edit-capable agent. Because "$@" is
# expanded last, any caller-supplied --chat-mode wins over this default.
exec "$AIDER_BIN" \
    --model "$MODEL" \
    --chat-mode ask \
    --no-auto-commits \
    --no-show-model-warnings \
    "$@"

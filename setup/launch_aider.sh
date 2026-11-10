#!/bin/bash
# ---------------------------------------------------------------------------
# launch_aider.sh -- Run aider configured for the SC26 tutorial vLLM server
#
# The tutorial's agent is Qwen Code (setup/launch_qwen.sh); this aider launcher
# is kept as an instructor fallback.
#
# Aider is a CLI coding agent. This wrapper points it at the locally-served
# Qwen3-Coder model from Module 7's vLLM backend, so no API keys or external
# accounts are needed. It runs aider from the tutorial's shared, read-only
# aider venv ($SC26_SHARED_DIR/sc26_aider_venv, built by the instructors with
# setup/shared-env/build_aider_venv.sh).
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
# Note:
# You need to start aider from the root of the git repo if you want it to read
# the AGENTS.md from the root directory. It will not read it if you start it 
# from a subdirectory.
#
# Type /help inside aider to see commands, /exit to quit.
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Pre-flight checks ------------------------------------------------------
# The aider install and the agent server URL file are both shared (under
# $SC26_SHARED_DIR), so every student runs the same agent against the same
# vLLM endpoint.
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
URL_FILE="$SC26_SHARED_DIR/sc26_agent/sc26_agent_server_url"
if [[ ! -f "$URL_FILE" ]]; then
    echo "ERROR: vLLM server URL file not found at $URL_FILE" >&2
    echo "       The agent backend isn't running. Ask an instructor for help." >&2
    exit 1
fi

# aider lives in its OWN venv (separate from the ML venv sc26_venv). We invoke
# its binary by absolute path so this never accidentally picks up some other
# aider on $PATH.
AIDER_VENV="$SC26_SHARED_DIR/sc26_aider_venv"
AIDER_BIN="$AIDER_VENV/bin/aider"
if [[ ! -x "$AIDER_BIN" ]]; then
    echo "ERROR: aider not found at $AIDER_BIN" >&2
    echo "       Please ask an instructor for help." >&2
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
# explains what to do, but never edits them. To let it edit, pass an edit
# format such as `--chat-mode diff`; `--chat-mode` is an alias of
# `--edit-format`, so `--chat-mode code` is rejected. Because "$@" is expanded
# last, any caller-supplied --chat-mode wins over this default.
exec "$AIDER_BIN" \
    --model "$MODEL" \
    --chat-mode ask \
    --no-auto-commits \
    --no-show-model-warnings \
    --no-check-update \
    "$@"

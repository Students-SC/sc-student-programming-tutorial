#!/bin/bash
# ---------------------------------------------------------------------------
# launch_qwen.sh -- Run Qwen Code configured for the SC26 tutorial vLLM server
#
# Qwen Code is a CLI coding agent: it can read files, search, run commands,
# and edit code on its own, asking for approval as it goes. This wrapper
# points it at the locally-served Qwen3-Coder model from the vLLM backend, so
# no API keys or external accounts are needed. It runs Qwen Code from the
# tutorial's shared, read-only install ($SC26_SHARED_DIR/sc26_qwen_code,
# built by the instructors with setup/shared-env/build_qwen_code.sh).
#
# By default Qwen Code starts in PLAN mode: it reads your files and explains,
# but does not edit them or run commands. Pass --approval-mode default to
# have it ask before each edit and command.
#
# Usage:
#     cd <project directory you want to work on>
#     bash <repo>/setup/launch_qwen.sh [qwen args...]
#
# Examples:
#     bash setup/launch_qwen.sh                            # plan mode
#     bash setup/launch_qwen.sh --approval-mode default    # ask before acting
#
# Press Shift+Tab inside Qwen Code to switch modes, type /help for commands,
# and /quit to exit.
# ---------------------------------------------------------------------------

set -euo pipefail

# --- Pre-flight checks ------------------------------------------------------
# The Qwen Code install and the agent server URL file are both shared (under
# $SC26_SHARED_DIR), so every student runs the same agent against the same
# vLLM endpoint.
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
URL_FILE="$SC26_SHARED_DIR/sc26_agent/sc26_agent_server_url"
if [[ ! -f "$URL_FILE" ]]; then
    echo "ERROR: vLLM server URL file not found at $URL_FILE" >&2
    echo "       The agent backend isn't running. Ask an instructor for help." >&2
    exit 1
fi

# Invoke the shared install by absolute path so this never picks up some
# other qwen on $PATH.
QWEN_DIR="$SC26_SHARED_DIR/sc26_qwen_code"
QWEN_BIN="$QWEN_DIR/bin/qwen"
if [[ ! -x "$QWEN_BIN" ]]; then
    echo "ERROR: Qwen Code not found at $QWEN_BIN" >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi

# --- Configure Qwen Code for the local vLLM server --------------------------
export OPENAI_BASE_URL="$(cat "$URL_FILE")"
export OPENAI_API_KEY="sk-no-key-needed"   # vLLM ignores this; Qwen Code needs *something*
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"

# The tutorial's defaults (plan mode, the server's context window, no
# auto-update) have the lowest precedence, so command-line flags and the
# participant's own settings override them. Qwen Code fails if a flag such
# as --approval-mode is given twice, so defaults must not be passed as flags.
export QWEN_CODE_SYSTEM_DEFAULTS_PATH="$QWEN_DIR/sc26-defaults.json"
export QWEN_USAGE_STATISTICS_ENABLED=0
export QWEN_TELEMETRY_ENABLED=0
export QWEN_CODE_MODELS_DEV=off

echo "=== Launching Qwen Code (SC26) ==="
echo "  API:    $OPENAI_BASE_URL"
echo "  Model:  $MODEL"
echo "  CWD:    $(pwd)"
echo ""
echo "Shift+Tab switches modes, /help lists commands, /quit exits."
echo ""

exec "$QWEN_BIN" --auth-type openai --model "$MODEL" "$@"

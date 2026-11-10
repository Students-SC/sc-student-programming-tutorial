#!/bin/bash
# ---------------------------------------------------------------------------
# start_server.sh -- Launch the shared vLLM inference server via Apptainer
#
# Lifecycle invariant maintained by this script:
#   The URL file ($URL_FILE) exists if and only if a vLLM server is running
#   AND ready to accept requests.
#
# To enforce that, the script:
#   1. Refuses to start if a URL file already exists (would mean another
#      server is already running).
#   2. Launches vLLM in the background.
#   3. Polls /v1/models until the server responds, then writes the URL file.
#   4. Removes the URL file on exit (success, scancel, timeout, or error)
#      via a trap.
#
# The site's job-submission filter caps mi3001x jobs at 4 hours, so on tutorial
# day the server runs as two chained jobs in the instructor reservation, with
# the second starting automatically when the first ends over lunch (see
# README.md, "Tutorial Day"):
#
#   JOB1=$(sbatch --parsable --reservation=sc26agent setup/vllm-server/start_server.sh)
#   sbatch --reservation=sc26agent --dependency=afterany:$JOB1 setup/vllm-server/start_server.sh
#
# Submit:  sbatch setup/vllm-server/start_server.sh
# Watch:   tail -f vllm-server_<JOBID>.out
# Stop:    scancel --name=vllm-server
# ---------------------------------------------------------------------------
#SBATCH --job-name=vllm-server
#SBATCH --qos=alloc_sc26dev_02232026_12312026
#SBATCH --partition=mi3001x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --output=vllm-server_%j.out
#SBATCH --error=vllm-server_%j.err

set -euo pipefail

# Create new files group-writable and group-only (no "other" access) so any
# member of the sc26dev group can manage them. This pairs with the setgid bit
# on $AGENT_DIR, which makes new files inherit the sc26dev group. The URL file
# is the one exception: students must read it, so it is published with "other"
# read permission (see "Publish the URL" below).
umask 0007

# --- Configuration ----------------------------------------------------------
# Shared tutorial resources live under $SC26_SHARED_DIR, which only the
# proj_sc26students group (students and instructors) can enter. The agent
# server's files -- container SIF, model cache, and the URL file -- live in its
# sc26_agent/ subdirectory, which is writable by the sc26dev group so any
# instructor can restart the server. Override SC26_SHARED_DIR to test against
# a different location.
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
AGENT_DIR="$SC26_SHARED_DIR/sc26_agent"

SIF_FILE="$AGENT_DIR/sc26_containers/vllm-openai-rocm.sif"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"
CACHE_DIR="$AGENT_DIR/sc26_model_cache"
PORT=8321
# Agents that call tools natively (Qwen Code) need vLLM to parse the model's
# tool calls. aider and Module 7's simple agents don't send tools, so they
# are unaffected. To try this model's other parser:
#   TOOL_CALL_PARSER=qwen3_coder sbatch setup/vllm-server/start_server.sh
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_xml}"
URL_FILE="$AGENT_DIR/sc26_agent_server_url"
READY_TIMEOUT=900   # max seconds to wait for vLLM to become ready

# --- Helpers ----------------------------------------------------------------
log() {
    echo "[$(date +%H:%M:%S)] $*"
}

# --- Pre-flight checks ------------------------------------------------------
log "=== Pre-flight checks ==="

if [[ ! -f "$SIF_FILE" ]]; then
    log "ERROR: Container image not found at $SIF_FILE"
    log "       Run 'sbatch setup/vllm-server/pull_image.sh' first."
    exit 1
fi

if [[ -e "$URL_FILE" ]]; then
    log "ERROR: A server URL file already exists at $URL_FILE"
    log "       This means a vLLM server is (or was) already running."
    log "       To stop an existing server:  scancel --name=vllm-server"
    log "       If you're sure no server is running, remove the file manually."
    exit 1
fi

mkdir -p "$CACHE_DIR"

# --- Server info ------------------------------------------------------------
NODE_HOSTNAME=$(hostname -s)
NODE_IP=$(hostname -I | awk '{print $1}')
SERVER_URL="http://${NODE_IP}:${PORT}/v1"

log "=== vLLM Inference Server (Apptainer) ==="
log "Node:         $NODE_HOSTNAME ($NODE_IP)"
log "Model:        $MODEL_NAME"
log "Port:         $PORT"
log "Tool parser:  $TOOL_CALL_PARSER"
log "URL (target): $SERVER_URL"
log "URL file:     $URL_FILE"
log "Container:    $SIF_FILE"
log "Cache:        $CACHE_DIR"
log ""

# --- Launch vLLM in the background -----------------------------------------
log "[STARTING] Launching vLLM container in background..."

# Initial trap: ensure URL file is removed on any exit.
# We'll update this once we have the vLLM PID so we also kill the server.
trap 'log "[CLEANUP] Removing URL file"; rm -f "$URL_FILE"' EXIT

apptainer run \
    --rocm \
    --bind "$CACHE_DIR:/model_cache" \
    --env "HF_HOME=/model_cache" \
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
    "$SIF_FILE" \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --download-dir /model_cache \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_CALL_PARSER" \
    --trust-remote-code &

VLLM_PID=$!
log "[STARTING] vLLM PID: $VLLM_PID"

# Update trap: also stop vLLM on exit.
trap 'log "[CLEANUP] Stopping vLLM (PID $VLLM_PID)"; kill -TERM "$VLLM_PID" 2>/dev/null || true; rm -f "$URL_FILE"' EXIT

# --- Wait for the server to be ready ---------------------------------------
log "[LOADING MODEL] Waiting for vLLM to become ready (timeout: ${READY_TIMEOUT}s)..."

START_TIME=$SECONDS
while true; do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        log "ERROR: vLLM process exited before becoming ready."
        log "       Check the logs above (and the .err file) for details."
        exit 1
    fi

    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        log "[READY] vLLM is responding to requests."
        break
    fi

    elapsed=$(( SECONDS - START_TIME ))
    if (( elapsed > READY_TIMEOUT )); then
        log "ERROR: Timed out after ${READY_TIMEOUT}s waiting for vLLM."
        exit 1
    fi

    sleep 5
done

# --- Publish the URL --------------------------------------------------------
# Students are not in sc26dev, so they read this file through its "other"
# permission bits. Nobody outside proj_sc26students can reach it, because
# $SC26_SHARED_DIR grants no "other" access.
(umask 0002; echo "$SERVER_URL" > "$URL_FILE")
log "[READY] Server URL written to $URL_FILE"
log "[READY] Server reachable at $SERVER_URL"
log ""
TIME_LIMIT=$(squeue -h -j "${SLURM_JOB_ID:-}" -o %l 2>/dev/null || true)
log "Server will run until the job ends (time limit: ${TIME_LIMIT:-unknown})."
log "To stop early: scancel --name=vllm-server"
log ""

# --- Wait for vLLM to terminate --------------------------------------------
wait "$VLLM_PID"

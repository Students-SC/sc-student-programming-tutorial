#!/bin/bash
# ---------------------------------------------------------------------------
# submit_agent.sh -- Run the agent exercise on a compute node
#
# Submit:  sbatch submit_agent.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=agent
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=agent_%j.out
#SBATCH --error=agent_%j.err

echo "AI Agent Exercise -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set. Ask an instructor for help." >&2
    exit 1
fi
VENV_DIR="$WORK/sc26_venv"
source "$VENV_DIR/bin/activate"

# Auto-discover the shared vLLM server URL from the instructor-provided file.
# The server writes its URL to this file once it's ready to accept requests.
SERVER_URL_FILE="$WORK/sc26_agent_server_url"

if [ -f "$SERVER_URL_FILE" ]; then
    export AGENT_API_URL=$(cat "$SERVER_URL_FILE")
else
    echo "ERROR: Server URL file not found at $SERVER_URL_FILE"
    echo "The vLLM server is not running. Ask an instructor for help."
    exit 1
fi
export AGENT_MODEL="${AGENT_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"

echo "API URL: $AGENT_API_URL"
echo "Model:   $AGENT_MODEL"
echo ""

# Run the agent -- change the script and task as needed
python3 build_agent.py \
    "Check what GPU is on this node, then write a simple HIP program that prints the GPU name from device code, compile it with hipcc, and run it."

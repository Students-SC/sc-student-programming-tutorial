#!/bin/bash
# ---------------------------------------------------------------------------
# download_model.sh -- Pre-download model weights to shared storage
#
# Run ONCE before the tutorial to avoid 80+ students downloading at once.
# Uses the vLLM container to ensure model cache compatibility.
#
# Prerequisites:
#   - Run pull_image.sh first to download the container SIF
#
# Submit:  sbatch setup/vllm-server/download_model.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=download-model
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH --output=download-model_%j.out
#SBATCH --error=download-model_%j.err

set -euo pipefail

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set." >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi

SIF_FILE="$WORK/sc26_containers/vllm-openai-rocm.sif"
MODEL_NAME="Qwen/Qwen3-Coder-30B-A3B-Instruct"
CACHE_DIR="$WORK/sc26_model_cache"

if [[ ! -f "$SIF_FILE" ]]; then
    echo "ERROR: Container image not found at $SIF_FILE"
    echo "Run 'sbatch setup/vllm-server/pull_image.sh' first."
    exit 1
fi

mkdir -p "$CACHE_DIR"

echo "=== Downloading model weights ==="
echo "Model:     $MODEL_NAME"
echo "Cache:     $CACHE_DIR"
echo "Container: $SIF_FILE"
echo "Date:      $(date)"
echo ""

apptainer exec \
    --bind "$CACHE_DIR:/model_cache" \
    --env "HF_HOME=/model_cache" \
    --env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}" \
    "$SIF_FILE" \
    python3 -c "
from huggingface_hub import snapshot_download

model_name = '$MODEL_NAME'
print(f'Downloading {model_name} to /model_cache...')
print('(~60 GB; this may take 15-30 minutes depending on network speed.)')
local_dir = snapshot_download(
    repo_id=model_name,
    cache_dir='/model_cache',
)
print(f'Download complete. Cached at: {local_dir}')
"

echo ""
echo "=== Download complete ==="
echo "Cache size:"
du -sh "$CACHE_DIR"

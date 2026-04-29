#!/bin/bash
# ---------------------------------------------------------------------------
# pull_image.sh -- Pull the vLLM ROCm Docker image via Apptainer
#
# This creates a Singularity/Apptainer SIF file (~10 GB) from the official
# vLLM ROCm Docker image on Docker Hub.
#
# Submit:  sbatch setup/vllm-server/pull_image.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=pull-vllm-sif
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=60:00
#SBATCH --output=pull-vllm-sif_%j.out
#SBATCH --error=pull-vllm-sif_%j.err

set -euo pipefail

if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set." >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi

SIF_DIR="$WORK/sc26_containers"
SIF_FILE="$SIF_DIR/vllm-openai-rocm.sif"

mkdir -p "$SIF_DIR"

# Use /work1 for the build cache (compute node /tmp may be too small for ~30 GB unpack)
export APPTAINER_CACHEDIR="$SIF_DIR/.apptainer_cache"
export APPTAINER_TMPDIR="$SIF_DIR/.apptainer_tmp"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

echo "=== Pulling vLLM ROCm Docker image ==="
echo "Source: docker://vllm/vllm-openai-rocm:latest"
echo "Target: $SIF_FILE"
echo "Date:   $(date)"
echo ""

#apptainer pull --force "$SIF_FILE" docker://vllm/vllm-openai-rocm:latest
apptainer pull "$SIF_FILE" docker://vllm/vllm-openai-rocm:latest

echo ""
echo "=== Pull complete ==="
echo "SIF file: $SIF_FILE"
ls -lh "$SIF_FILE"
echo ""
echo "Date: $(date)"

# Clean up build cache (can be large)
rm -rf "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

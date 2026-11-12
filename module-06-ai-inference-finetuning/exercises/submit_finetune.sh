#!/bin/bash
# ---------------------------------------------------------------------------
# submit_finetune.sh -- Run LoRA fine-tuning on a GPU node
#
# Submit:  sbatch submit_finetune.sh [script.py]
# ---------------------------------------------------------------------------
#SBATCH --job-name=finetune
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

set -euo pipefail

echo "LoRA Fine-Tuning -- Job ${SLURM_JOB_ID:-(not in Slurm)} on $(hostname)"
echo ""

# The shared, read-only Python environment that the instructors built
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
VENV_DIR="$SC26_SHARED_DIR/sc26_venv"
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: shared Python environment not found at $VENV_DIR" >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi
source "$VENV_DIR/bin/activate"

# Load models from the shared, read-only Hugging Face cache, without
# contacting the Hub (see setup/shared-env/README.md).
export HF_HUB_CACHE="$SC26_SHARED_DIR/sc26_hf_cache/hub"
export HF_HUB_OFFLINE=1

# The way these scripts load the dataset, the datasets library must be able to
# write next to it, so copy the prepared dataset from the read-only shared
# cache into $WORK once and use the copy. A copy counts as complete only if it
# contains the marker file, which is written after everything else succeeded.
if [[ -z "${WORK:-}" ]]; then
    echo "ERROR: The WORK environment variable is not set. Ask an instructor for help." >&2
    exit 1
fi
export HF_DATASETS_CACHE="$WORK/.cache/sc26_hf_datasets"
DATASET_DIR="$HF_DATASETS_CACHE/imdb"
MARKER="$DATASET_DIR/.sc26_copy_complete"
fail_copy() {
    echo "ERROR: could not prepare the IMDB dataset in $HF_DATASETS_CACHE: $1" >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
}
if [[ ! -f "$MARKER" ]]; then
    if [[ -e "$DATASET_DIR" ]]; then
        echo "ERROR: $DATASET_DIR exists but is not a complete copy." >&2
        echo "       Remove it with: rm -rf \"$DATASET_DIR\"" >&2
        echo "       and submit the job again." >&2
        exit 1
    fi
    echo "Copying the IMDB dataset to $HF_DATASETS_CACHE (first run only)..."
    mkdir -p "$HF_DATASETS_CACHE" || fail_copy "cannot create the directory"
    TMP_COPY=$(mktemp -d "$HF_DATASETS_CACHE/imdb.partial.XXXXXX") || fail_copy "cannot create a temporary directory"
    trap 'rm -rf "$TMP_COPY"' EXIT
    cp -r "$SC26_SHARED_DIR/sc26_hf_cache/datasets/imdb/." "$TMP_COPY/" || fail_copy "copy failed"
    chmod -R u+w "$TMP_COPY" || fail_copy "cannot make the copy writable"
    touch "$TMP_COPY/.sc26_copy_complete" || fail_copy "cannot write the marker file"
    # mv -T fails, without replacing anything, if another of your jobs
    # published its copy first; then use that copy if it is complete.
    if ! mv -T "$TMP_COPY" "$DATASET_DIR" 2>/dev/null && [[ ! -f "$MARKER" ]]; then
        fail_copy "cannot move the finished copy into place"
    fi
fi

echo "Python:  $(python3 --version)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo ""

# The Python script to run is the first argument (default: the exercise)
SCRIPT="${1:-finetune_lora.py}"
if [[ ! -f "$SCRIPT" ]]; then
    echo "ERROR: Python script not found: $SCRIPT" >&2
    exit 1
fi
echo "Running: $SCRIPT"
echo ""
python3 "$SCRIPT"

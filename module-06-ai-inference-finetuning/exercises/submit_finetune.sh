#!/bin/bash
# ---------------------------------------------------------------------------
# submit_finetune.sh -- Run LoRA fine-tuning on a GPU node
#
# Submit:  sbatch submit_finetune.sh [script.py]
# ---------------------------------------------------------------------------
#SBATCH --job-name=finetune
#SBATCH -A TRN001 --reservation=sc26studentprogram --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10:00
#SBATCH --output=finetune_%j.out
#SBATCH --error=finetune_%j.err

set -euo pipefail

echo "LoRA Fine-Tuning -- Job ${SLURM_JOB_ID:-(not in Slurm)} on $(hostname)"
echo ""

module load rocm/7.2
module load miniforge3
# The shared, read-only Python environment that the instructors built
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/lustre/orion/stf007/world-shared/subil/SC26_student_program}"
VENV_DIR="$SC26_SHARED_DIR/sc26_venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "ERROR: shared Python environment not found at $VENV_DIR" >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
fi
conda activate $VENV_DIR

# Load models from the shared, read-only Hugging Face cache, without
# contacting the Hub (see setup/shared-env/README.md).
export HF_HOME=$SC26_SHARED_DIR/sc26_hf_cache
export HF_HUB_CACHE="$SC26_SHARED_DIR/sc26_hf_cache/hub"
export HF_HUB_OFFLINE=1

# The way these scripts load the dataset, the datasets library must be able to
# write next to it, so copy the prepared dataset from the read-only shared
# cache into your parallel filesystem location once and use the copy. A copy counts as complete only if it
# contains the marker file, which is written after everything else succeeded.

PROJ=$(sacctmgr -nP show assoc where user=$USER format=account | head -n1)
export HF_DATASETS_CACHE="$MEMBERWORK/$PROJ/.cache/sc26_hf_datasets"
fail_copy() {
    echo "ERROR: could not prepare the IMDB dataset in $HF_DATASETS_CACHE: $1" >&2
    echo "       Please ask an instructor for help." >&2
    exit 1
}

    echo "Copying the IMDB dataset to $HF_DATASETS_CACHE (first run only)..."
    mkdir -p "$HF_DATASETS_CACHE" || fail_copy "cannot create the directory"

    cp -r "$SC26_SHARED_DIR/sc26_hf_cache/datasets/." $HF_DATASETS_CACHE || fail_copy "copy failed"
    chmod -R u+w $HF_DATASETS_CACHE || fail_copy "cannot make the copy writable"
    

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

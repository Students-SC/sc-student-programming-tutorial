#!/bin/bash
# ---------------------------------------------------------------------------
# download_hf_cache.sh -- Pre-download Module 6's models and dataset
#
# INSTRUCTOR-ONLY. Downloads the models and dataset that Module 6 uses into a
# shared Hugging Face cache, so 80-100 students don't each download them from
# the Hub (unauthenticated, from the same few IP addresses) during the
# tutorial. Module 6's jobs then load everything offline:
#
#   sc26_hf_cache/hub/       models and the dataset's raw files, which
#                            students load read-only (HF_HUB_CACHE)
#   sc26_hf_cache/datasets/  the dataset, already prepared; each student's
#                            job copies it to $WORK once, because the datasets
#                            library must be able to write to its cache
#
# Uses the shared ML venv, so run build_venv.sh first. Only the owner of
# $SC26_SHARED_DIR can write there, so the owner must run this.
#
# Submit (from the repo root):  sbatch setup/shared-env/download_hf_cache.sh
# If mi2101x is busy:           sbatch -p devel setup/shared-env/download_hf_cache.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=download-hf-cache
#SBATCH --qos=alloc_sc26dev_02232026_12312026
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --output=download-hf-cache_%j.out
#SBATCH --error=download-hf-cache_%j.err

set -euo pipefail

# Group-readable, never group-writable. See setup/shared-env/README.md.
umask 0027

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
CACHE_DIR="$SC26_SHARED_DIR/sc26_hf_cache"
PYTHON="$SC26_SHARED_DIR/sc26_venv/bin/python"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Pre-flight checks ------------------------------------------------------
if [[ ! -w "$SC26_SHARED_DIR" ]]; then
    log "ERROR: $SC26_SHARED_DIR is not writable by $USER."
    log "       Only its owner ($(stat -c %U "$SC26_SHARED_DIR")) can build the shared cache."
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    log "ERROR: shared venv not found at $SC26_SHARED_DIR/sc26_venv."
    log "       Run 'sbatch setup/shared-env/build_venv.sh' first."
    exit 1
fi

export HF_HUB_CACHE="$CACHE_DIR/hub"
export HF_DATASETS_CACHE="$CACHE_DIR/datasets"
# Keep Hugging Face's other files (token, download-accelerator cache) out of
# the shared tree.
HF_HOME="$(mktemp -d)"
export HF_HOME
trap 'rm -rf "$HF_HOME"' EXIT
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

log "=== Downloading Module 6's models and dataset ==="
log "Hub cache:      $HF_HUB_CACHE"
log "Datasets cache: $HF_DATASETS_CACHE"

# Loading each model with transformers downloads exactly the files it needs.
# Keep these names in sync with MODEL_NAME in the Module 6 scripts.
"$PYTHON" - <<'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset

AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", dtype=torch.float16)
print("TinyLlama/TinyLlama-1.1B-Chat-v1.0: downloaded")

AutoTokenizer.from_pretrained("distilbert-base-uncased")
AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
print("distilbert-base-uncased: downloaded")

ds = load_dataset("imdb")
print("imdb: downloaded and prepared,", {split: len(rows) for split, rows in ds.items()})
EOF

# --- Checks -----------------------------------------------------------------
log "=== Checking the cache ==="
HF_HUB_OFFLINE=1 "$PYTHON" - <<'EOF'
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
AutoTokenizer.from_pretrained("distilbert-base-uncased")
load_dataset("imdb")
print("Offline load OK: both models and the dataset")
EOF

not_group_readable=$(find "$CACHE_DIR" ! -type l ! -perm -g=r | head -5)
group_writable=$(find "$CACHE_DIR" ! -type l -perm /g=w | head -5)
if [[ -n "$not_group_readable" || -n "$group_writable" ]]; then
    log "ERROR: unexpected permissions (students need read-only access):"
    echo "$not_group_readable" "$group_writable"
    exit 1
fi
log "Permissions OK: group-readable, not group-writable."
log "=== Done ==="
du -sh "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

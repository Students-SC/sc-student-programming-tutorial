#!/bin/bash
# ---------------------------------------------------------------------------
# build_venv.sh -- Build the shared, read-only Python venv for the AI modules
#
# INSTRUCTOR-ONLY. Builds one venv with PyTorch (ROCm), transformers, and the
# other packages pinned in requirements-sc26-venv.txt, under $SC26_SHARED_DIR,
# where every student can use it but nobody else can change it. Only the
# owner of $SC26_SHARED_DIR can write there, so the owner must run this.
#
# Each build goes into a new timestamped directory (sc26_venv-YYYYMMDD-HHMMSS).
# After it passes its checks, the sc26_venv symlink is switched to it in one
# step, so rebuilding never breaks jobs that are using the current venv, and
# the previous build remains for rollback.
#
# Submit (from the repo root):  sbatch setup/shared-env/build_venv.sh
# If mi2101x is busy:           sbatch -p devel setup/shared-env/build_venv.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=build-sc26-venv
#SBATCH --qos=alloc_sc26dev_02232026_12312026
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=30:00
#SBATCH --output=build-sc26-venv_%j.out
#SBATCH --error=build-sc26-venv_%j.err

set -euo pipefail

# Group-readable, never group-writable: students (the group) can use the venv
# but cannot modify it. See setup/shared-env/README.md.
umask 0027

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/setup/shared-env"
REQUIREMENTS="$SCRIPT_DIR/requirements-sc26-venv.txt"
PYTHON=/usr/bin/python3.12
STAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_DIR="$SC26_SHARED_DIR/sc26_venv-$STAMP"
LINK="$SC26_SHARED_DIR/sc26_venv"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Pre-flight checks ------------------------------------------------------
if [[ ! -f "$REQUIREMENTS" ]]; then
    log "ERROR: $REQUIREMENTS not found. Submit from the repo root."
    exit 1
fi
if [[ ! -w "$SC26_SHARED_DIR" ]]; then
    log "ERROR: $SC26_SHARED_DIR is not writable by $USER."
    log "       Only its owner ($(stat -c %U "$SC26_SHARED_DIR")) can build the shared venv."
    exit 1
fi

log "=== Building the shared ML venv ==="
log "Node:          $(hostname)"
log "Target:        $BUILD_DIR"
log "Requirements:  $REQUIREMENTS"

# --- Build ------------------------------------------------------------------
"$PYTHON" -m venv "$BUILD_DIR"
"$BUILD_DIR/bin/python" -m pip install --no-cache-dir --quiet --upgrade pip
# --no-cache-dir keeps multi-GB wheels out of the small $HOME quota.
"$BUILD_DIR/bin/python" -m pip install --no-cache-dir -r "$REQUIREMENTS"

# Students can't write __pycache__ in a read-only venv, so make sure every
# module is already compiled; otherwise each import would recompile it.
"$BUILD_DIR/bin/python" -m compileall -q "$BUILD_DIR/lib" > /dev/null \
    || log "Note: a few files did not compile (normal for test data shipped in some packages)."

# --- Checks -----------------------------------------------------------------
log "=== Checking the build ==="
"$BUILD_DIR/bin/python" - <<'EOF'
import torch, transformers, datasets, peft, accelerate
print(f"PyTorch {torch.__version__}, transformers {transformers.__version__}, "
      f"datasets {datasets.__version__}, peft {peft.__version__}")
assert torch.cuda.is_available(), "PyTorch cannot see the GPU"
print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

not_group_readable=$(find "$BUILD_DIR" ! -type l ! -perm -g=r | head -5)
group_writable=$(find "$BUILD_DIR" ! -type l -perm /g=w | head -5)
if [[ -n "$not_group_readable" || -n "$group_writable" ]]; then
    log "ERROR: unexpected permissions (students need read-only access):"
    echo "$not_group_readable" "$group_writable"
    exit 1
fi
log "Permissions OK: group-readable, not group-writable."

# --- Publish ------------------------------------------------------------------
# Switch the symlink atomically (rename over the old link).
ln -s "$(basename "$BUILD_DIR")" "$LINK.new.$STAMP"
mv -T "$LINK.new.$STAMP" "$LINK"
log "=== Done: $LINK -> $(readlink "$LINK") ==="
log "Size: $(du -sh "$BUILD_DIR" | cut -f1)"
log "Older builds (remove when no longer needed):"
find "$SC26_SHARED_DIR" -maxdepth 1 -name 'sc26_venv-*' ! -path "$BUILD_DIR" -printf '  %p\n'

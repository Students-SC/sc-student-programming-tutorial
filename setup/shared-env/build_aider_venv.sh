#!/bin/bash
# ---------------------------------------------------------------------------
# build_aider_venv.sh -- Build the shared, read-only aider (CLI agent) venv
#
# INSTRUCTOR-ONLY. aider is kept in its own venv, separate from the ML venv,
# because it pins older openai/pydantic versions than the PyTorch/transformers
# stack. Only the owner of $SC26_SHARED_DIR can write there, so the owner must
# run this.
#
# This is a pure pip install with no GPU or compile step, so it runs on the
# login node (no Slurm job). As with build_venv.sh, each build goes into a new
# timestamped directory and the sc26_aider_venv symlink is switched to it in
# one step after its checks pass.
#
# Run:  bash setup/shared-env/build_aider_venv.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Group-readable, never group-writable: students (the group) can run aider
# but cannot modify its install. See setup/shared-env/README.md.
umask 0027

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
# aider-chat and every dependency are pinned, so each rebuild gives students
# the same agent.
REQUIREMENTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/requirements-sc26-aider-venv.txt"
PYTHON=/usr/bin/python3.12
STAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_DIR="$SC26_SHARED_DIR/sc26_aider_venv-$STAMP"
LINK="$SC26_SHARED_DIR/sc26_aider_venv"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Pre-flight checks ------------------------------------------------------
if [[ ! -f "$REQUIREMENTS" ]]; then
    log "ERROR: $REQUIREMENTS not found."
    exit 1
fi
if [[ ! -w "$SC26_SHARED_DIR" ]]; then
    log "ERROR: $SC26_SHARED_DIR is not writable by $USER."
    log "       Only its owner ($(stat -c %U "$SC26_SHARED_DIR")) can build the shared aider venv."
    exit 1
fi

log "=== Building the shared aider venv ==="
log "Host:          $(hostname)"
log "Target:        $BUILD_DIR"
log "Requirements:  $REQUIREMENTS"

# --- Build ------------------------------------------------------------------
"$PYTHON" -m venv "$BUILD_DIR"
"$BUILD_DIR/bin/python" -m pip install --no-cache-dir --quiet --upgrade pip
"$BUILD_DIR/bin/python" -m pip install --no-cache-dir --quiet -r "$REQUIREMENTS"

# Students can't write __pycache__ in a read-only venv; precompile everything.
"$BUILD_DIR/bin/python" -m compileall -q "$BUILD_DIR/lib" > /dev/null \
    || log "Note: a few files did not compile (normal for test data shipped in some packages)."

# --- Checks -----------------------------------------------------------------
log "=== Checking the build ==="
"$BUILD_DIR/bin/aider" --version

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
find "$SC26_SHARED_DIR" -maxdepth 1 -name 'sc26_aider_venv-*' ! -path "$BUILD_DIR" -printf '  %p\n'

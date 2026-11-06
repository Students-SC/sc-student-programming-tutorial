#!/bin/bash
# ---------------------------------------------------------------------------
# build_qwen_code.sh -- Install the shared, read-only Qwen Code (CLI agent)
#
# INSTRUCTOR-ONLY. Qwen Code ships as a standalone archive that bundles its
# own Node.js runtime, so this downloads a pinned release, verifies its
# checksum, and unpacks it -- no npm or system Node.js is needed. Only the
# owner of $SC26_SHARED_DIR can write there, so the owner must run this.
#
# It runs on the login node (no Slurm job). As with the venvs, each install
# goes into a new timestamped directory and the sc26_qwen_code symlink is
# switched to it in one step after its checks pass.
#
# The install also gets a copy of qwen-code-defaults.json, which
# setup/launch_qwen.sh passes to Qwen Code as its lowest-precedence settings
# (plan mode by default, no auto-update or usage statistics, and the
# server's 32K context window). Re-run this script after editing that file.
#
# Run:  bash setup/shared-env/build_qwen_code.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Group-readable, never group-writable: students (the group) can run Qwen
# Code but cannot modify its install. See setup/shared-env/README.md.
umask 0027

SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
# Qwen Code releases almost daily; pin one so every student gets the same
# agent, and change it only after re-testing.
QWEN_CODE_VERSION="0.26.8"
RELEASE_URL="https://github.com/QwenLM/qwen-code/releases/download/v$QWEN_CODE_VERSION"
ARCHIVE="qwen-code-linux-x64.tar.gz"
DEFAULTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen-code-defaults.json"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUILD_DIR="$SC26_SHARED_DIR/sc26_qwen_code-$STAMP"
LINK="$SC26_SHARED_DIR/sc26_qwen_code"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# --- Pre-flight checks ------------------------------------------------------
if [[ ! -f "$DEFAULTS" ]]; then
    log "ERROR: $DEFAULTS not found."
    exit 1
fi
if [[ ! -w "$SC26_SHARED_DIR" ]]; then
    log "ERROR: $SC26_SHARED_DIR is not writable by $USER."
    log "       Only its owner ($(stat -c %U "$SC26_SHARED_DIR")) can install the shared Qwen Code."
    exit 1
fi

log "=== Installing the shared Qwen Code ==="
log "Host:     $(hostname)"
log "Version:  $QWEN_CODE_VERSION"
log "Target:   $BUILD_DIR"

# --- Download and verify ----------------------------------------------------
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
curl -fsSL -o "$TMP_DIR/$ARCHIVE" "$RELEASE_URL/$ARCHIVE"
curl -fsSL -o "$TMP_DIR/SHA256SUMS" "$RELEASE_URL/SHA256SUMS"
(cd "$TMP_DIR" && grep " $ARCHIVE\$" SHA256SUMS | sha256sum --check --quiet)
log "Checksum OK."

# --- Install ----------------------------------------------------------------
mkdir "$BUILD_DIR"
tar -xzf "$TMP_DIR/$ARCHIVE" -C "$BUILD_DIR" --strip-components=1
cp "$DEFAULTS" "$BUILD_DIR/sc26-defaults.json"

# --- Checks -----------------------------------------------------------------
log "=== Checking the install ==="
installed="$("$BUILD_DIR/bin/qwen" --version)"
if [[ "$installed" != "$QWEN_CODE_VERSION" ]]; then
    log "ERROR: expected Qwen Code $QWEN_CODE_VERSION, got '$installed'."
    exit 1
fi
log "Qwen Code $installed"

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
log "Older installs (remove when no longer needed):"
find "$SC26_SHARED_DIR" -maxdepth 1 -name 'sc26_qwen_code-*' ! -path "$BUILD_DIR" -printf '  %p\n'

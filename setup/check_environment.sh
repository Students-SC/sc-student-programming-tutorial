#!/bin/bash
# ---------------------------------------------------------------------------
# check_environment.sh -- Verify that the tutorial environment is ready
#
# Run on the LOGIN node:  bash setup/check_environment.sh
# ---------------------------------------------------------------------------

PASS=0
FAIL=0

check() {
    local description="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  [PASS]  $description"
        ((PASS++))
    else
        echo "  [FAIL]  $description"
        ((FAIL++))
    fi
}

echo ""
echo "============================================="
echo "  SC26 Tutorial -- Environment Check"
echo "============================================="
echo ""

# ---- Cluster access ----
echo "--- Cluster Access ---"
check "Login node reachable (you're here!)" true
check "Home directory exists" test -d "$HOME"

# ---- Modules ----
echo ""
echo "--- Software Modules ---"
check "module command available" type module
check "hpcfund module loaded" bash -c 'module list 2>&1 | grep -q hpcfund'
check "rocm module loaded" bash -c 'module list 2>&1 | grep -q rocm'
check "gnu12 module loaded" bash -c 'module list 2>&1 | grep -q gnu12'
check "openmpi4 module loaded" bash -c 'module list 2>&1 | grep -q openmpi4'

# ---- Compilers & tools ----
echo ""
echo "--- Compilers & Tools ---"
check "gcc available" which gcc
check "g++ available" which g++
check "mpicc available" which mpicc
check "mpicxx available" which mpicxx
check "hipcc available" which hipcc
check "cmake available" which cmake
check "make available" which make
check "python3 available" which python3

# ---- Slurm ----
echo ""
echo "--- Slurm Scheduler ---"
check "sbatch available" which sbatch
check "srun available" which srun
check "squeue available" which squeue
check "sinfo available" which sinfo
check "mi2101x partition visible" bash -c 'sinfo -p mi2101x 2>&1 | grep -q mi2101x'

# ---- Tutorial environment ----
echo ""
echo "--- Tutorial Environment ---"
SC26_SHARED_DIR="${SC26_SHARED_DIR:-/work1/proj_sc26students/shared}"
check "WORK directory set and present" test -d "${WORK:-/nonexistent}"
check "Shared Python environment readable" test -r "$SC26_SHARED_DIR/sc26_venv/bin/activate"
check "Shared AI agent (aider) installed" test -x "$SC26_SHARED_DIR/sc26_aider_venv/bin/aider"
check "Shared model cache readable" test -r "$SC26_SHARED_DIR/sc26_hf_cache/hub"
# The agent server may not be running yet, so this is a warning, not a failure.
if [[ -r "$SC26_SHARED_DIR/sc26_agent/sc26_agent_server_url" ]]; then
    echo "  [PASS]  AI agent server running"
    ((PASS++))
else
    echo "  [WARN]  AI agent server not running yet (instructors start it before the tutorial)"
fi

# ---- Summary ----
echo ""
echo "============================================="
echo "  Results:  $PASS passed,  $FAIL failed"
echo "============================================="

if [ "$FAIL" -gt 0 ]; then
    echo ""
    echo "  Some checks failed. Please ask an instructor for help."
    echo ""
    exit 1
else
    echo ""
    echo "  All checks passed -- you're ready to go!"
    echo ""
    exit 0
fi

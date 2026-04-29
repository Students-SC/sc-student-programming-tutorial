#!/bin/bash
# ---------------------------------------------------------------------------
# explore_cluster.sh -- Guided exploration of the AUP AI & HPC Cluster
#
# Run this on the LOGIN node:
#   bash explore_cluster.sh
#
# The script runs commands and pauses between sections so you can read
# the output. Press Enter to continue after each section.
# ---------------------------------------------------------------------------

pause() {
    echo ""
    echo "--- Press Enter to continue ---"
    read -r
    echo ""
}

section() {
    echo "============================================="
    echo "  $1"
    echo "============================================="
    echo ""
}

# -------------------------------------------------------
section "Step 1: Where Are You?"
# -------------------------------------------------------
echo "Hostname:          $(hostname)"
echo "Username:          $(whoami)"
echo "Home directory:    $HOME"
echo "Current directory: $(pwd)"
echo "Date/time:         $(date)"
pause

# -------------------------------------------------------
section "Step 2: Filesystem"
# -------------------------------------------------------
echo "--- Your home directory ---"
ls -la "$HOME" | head -20
echo ""
echo "--- Your work directory ---"
ls -la "$WORK" | head -20
echo ""
echo "--- home disk space ---"
df -h "$HOME" 2>/dev/null || echo "(df not available for this path)"
echo ""
echo "--- work disk space ---"
df -h "$WORK" 2>/dev/null || echo "(df not available for this path)"
pause

# -------------------------------------------------------
section "Step 3: Software Modules"
# -------------------------------------------------------
echo "--- Currently loaded modules ---"
module -t list 2>&1
echo ""
echo "--- Available modules ---"
module -t avail 2>&1
echo ""
echo "--- What does 'hpcfund' provide? ---"
module show hpcfund 2>&1
pause

# -------------------------------------------------------
section "Step 4: CPU Information"
# -------------------------------------------------------
echo "CPU model:         $(lscpu | grep 'Model name' | sed 's/.*:\s*//')"
echo "Sockets:           $(lscpu | grep 'Socket(s)' | sed 's/.*:\s*//')"
echo "Cores per socket:  $(lscpu | grep 'Core(s) per socket' | sed 's/.*:\s*//')"
echo "Total CPUs:        $(lscpu | grep '^CPU(s):' | sed 's/.*:\s*//')"
echo ""
echo "--- Memory ---"
free -h
echo ""
echo "NOTE: This is the LOGIN node. The compute nodes we'll use have 16 cores and 64 GB RAM."
pause

# -------------------------------------------------------
section "Step 5: GPU Software Stack"
# -------------------------------------------------------
echo "HIP compiler: $(which hipcc 2>/dev/null || echo 'not found')"
echo ""
if command -v hipcc &>/dev/null; then
    echo "--- hipcc version ---"
    hipcc --version 2>&1
fi
echo ""
echo "--- ROCm info (first 20 lines) ---"
rocminfo 2>/dev/null | head -20 || echo "(rocminfo not available on this node)"
echo ""
echo "NOTE: Full GPU details will be visible when running on a compute node (Module 2)."
pause

# -------------------------------------------------------
section "Step 6: The Cluster (Slurm)"
# -------------------------------------------------------
echo "--- All partitions ---"
sinfo 2>/dev/null || echo "(sinfo not available)"
echo ""
echo "--- Our partition (mi2101x) detail ---"
sinfo -p mi2101x -N -l 2>/dev/null || echo "(partition not found)"
echo ""
echo "--- Current job queue ---"
squeue 2>/dev/null | head -20 || echo "(squeue not available)"
pause

# -------------------------------------------------------
section "Exploration Complete!"
# -------------------------------------------------------
echo "You've seen:"
echo "  - Where you are (login node)"
echo "  - The filesystem layout"
echo "  - The software module system"
echo "  - CPU and memory information"
echo "  - The GPU software stack"
echo "  - The Slurm cluster overview"
echo ""
echo "Next up: Module 2 -- submitting your first job to a compute node!"
echo ""

#!/bin/bash
# ---------------------------------------------------------------------------
# explore_cluster.sh -- Guided exploration of the AUP AI & HPC Cluster
#
# Run this on the LOGIN node:
#   bash explore_cluster.sh
#
# For each step, you'll see a short description and the command to run.
# Type the command yourself at the prompt to build muscle memory.
# ---------------------------------------------------------------------------

section() {
    echo ""
    echo "============================================="
    echo "  $1"
    echo "============================================="
}

run_command() {
    local description="$1"
    local command="$2"

    echo ""
    echo "$description"
    echo ""
    echo "Type the command below exactly as shown:"
    echo "    $command"
    while true; do
        read -e -p "$ " user_input
        if [ "$user_input" = "$command" ]; then
            break
        elif [ -z "$user_input" ]; then
            continue
        else
            echo "Not quite -- try again. Expected: $command"
        fi
    done
    echo ""
    eval "$command"
    echo ""
}

# -------------------------------------------------------
section "Step 1: Where Are You?"
# -------------------------------------------------------
run_command "Print the name of the machine you are logged into." \
    "hostname"
run_command "Print your username on this system." \
    "whoami"
run_command "Print the path to your home directory (the \$HOME variable)." \
    "echo \$HOME"
run_command "Print the current working directory." \
    "pwd"
run_command "Print the current date and time." \
    "date"

# -------------------------------------------------------
section "Step 2: Filesystem"
# -------------------------------------------------------
run_command "List everything in your home directory, long format, including hidden files." \
    "ls -la \$HOME"
run_command "List everything in your work directory (the \$WORK area for larger files)." \
    "ls -la \$WORK"
run_command "Show disk space usage for the filesystem holding your home directory." \
    "df -h \$HOME"
run_command "Show disk space usage for the filesystem holding your work directory." \
    "df -h \$WORK"

# -------------------------------------------------------
section "Step 3: Software Modules"
# -------------------------------------------------------
run_command "List the environment modules currently loaded in your shell." \
    "module list"
run_command "List every module available to load on this cluster." \
    "module avail"
run_command "Show details about what the 'hpcfund' module sets up when loaded." \
    "module show hpcfund"

# -------------------------------------------------------
section "Step 4: CPU Information"
# -------------------------------------------------------
run_command "Print CPU architecture details: model, sockets, cores, threads." \
    "lscpu"
run_command "Show memory usage and total RAM in human-readable units." \
    "free -h"
echo ""
echo "NOTE: This is the LOGIN node. The compute nodes we'll use have 16 cores and 64 GB RAM."

# -------------------------------------------------------
section "Step 5: GPU Software Stack"
# -------------------------------------------------------
run_command "Find the path of the HIP compiler (AMD's equivalent of nvcc)." \
    "which hipcc"
run_command "Print the HIP compiler version." \
    "hipcc --version"
run_command "Show ROCm (AMD GPU runtime) info -- piped to 'head' since output is long." \
    "rocminfo | head -20"
echo ""
echo "NOTE: Full GPU details will be visible when running on a compute node (Module 2)."

# -------------------------------------------------------
section "Step 6: The Cluster (Slurm)"
# -------------------------------------------------------
run_command "Show all Slurm partitions and the state of their nodes." \
    "sinfo"
run_command "Show node-level details just for the mi2101x partition (the one we'll use)." \
    "sinfo -p mi2101x -N -l"
run_command "Show the current job queue across the cluster." \
    "squeue"

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
echo "The README.md file has all the commands you typed if you need to reference them later"
echo ""
echo "Next up: Module 2 -- submitting your first job to a compute node!"
echo ""

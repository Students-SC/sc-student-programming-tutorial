#!/bin/bash
# ---------------------------------------------------------------------------
# submit_hello.sh -- Run the hello_compute program on a compute node
#
# First compile:  gcc -o hello_compute hello_compute.c
# Then submit:    sbatch submit_hello.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=hello-compute
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00
#SBATCH --output=hello-compute_%j.out
#SBATCH --error=hello-compute_%j.err

echo "Running hello_compute on $(hostname)..."
echo ""

./hello_compute

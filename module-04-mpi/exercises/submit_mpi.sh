#!/bin/bash
# ---------------------------------------------------------------------------
# submit_mpi.sh -- Run parallel_sum with varying rank counts
#
# First compile:  mpicc -O2 -o parallel_sum parallel_sum.c -lm
# Then submit:    sbatch submit_mpi.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=mpi-sum
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=2:00
#SBATCH --output=mpi-sum_%j.out
#SBATCH --error=mpi-sum_%j.err

# Stop the sweep if a command fails, including any srun step.
set -euo pipefail

echo "MPI Parallel Sum Scaling -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

# Three ranks checks the remainder: 1 billion terms don't divide evenly by 3.
for RANKS in 1 2 3 4 8 16; do
    echo "=== $RANKS rank(s) ==="
    srun --ntasks=$RANKS ./parallel_sum
    echo ""
done

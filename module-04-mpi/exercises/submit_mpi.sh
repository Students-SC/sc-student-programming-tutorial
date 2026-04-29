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
#SBATCH --time=5:00
#SBATCH --output=mpi-sum_%j.out
#SBATCH --error=mpi-sum_%j.err

echo "MPI Parallel Sum Scaling -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

for RANKS in 1 2 4 8 16; do
    echo "=== $RANKS rank(s) ==="
    srun --ntasks=$RANKS ./parallel_sum
    echo ""
done

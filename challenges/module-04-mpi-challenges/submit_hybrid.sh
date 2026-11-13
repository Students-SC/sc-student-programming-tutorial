#!/bin/bash
# ---------------------------------------------------------------------------
# submit_hybrid.sh -- Run hybrid_sum with different ranks x threads splits
#
# First compile:  mpicc -fopenmp -O2 -o hybrid_sum hybrid_sum.c -lm
# Then submit:    sbatch submit_hybrid.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=hybrid-sum
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=5:00
#SBATCH --output=hybrid-sum_%j.out
#SBATCH --error=hybrid-sum_%j.err

# Stop the sweep if a command fails, including any srun step.
set -euo pipefail

echo "Hybrid MPI + OpenMP Sum -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

# Every split uses all 16 cores: ranks x threads = 16. --cpus-per-task gives
# each rank its own cores for its threads.
for RANKS in 16 4 2 1; do
    THREADS=$((16 / RANKS))
    echo "=== $RANKS rank(s) x $THREADS thread(s) ==="
    OMP_NUM_THREADS=$THREADS srun --ntasks=$RANKS --cpus-per-task=$THREADS ./hybrid_sum
    echo ""
done

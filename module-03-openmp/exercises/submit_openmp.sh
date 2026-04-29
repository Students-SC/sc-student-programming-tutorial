#!/bin/bash
# ---------------------------------------------------------------------------
# submit_openmp.sh -- Run the pi_openmp program with varying thread counts
#
# First compile:  gcc -fopenmp -O2 -o pi_openmp pi_openmp.c -lm
# Then submit:    sbatch submit_openmp.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=openmp-pi
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00
#SBATCH --output=openmp-pi_%j.out
#SBATCH --error=openmp-pi_%j.err

echo "OpenMP Pi Scaling -- Job $SLURM_JOB_ID on $(hostname)"
echo "Available CPUs: $SLURM_CPUS_PER_TASK"
echo ""

for THREADS in 1 2 4 8 16; do
    echo "=== OMP_NUM_THREADS=$THREADS ==="
    export OMP_NUM_THREADS=$THREADS
    ./pi_openmp
    echo ""
done

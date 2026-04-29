#!/bin/bash
# ---------------------------------------------------------------------------
# parameter_sweep.sh -- Run a computation for multiple parameter values
#
# CHALLENGE EXERCISE: Fill in the TODO section below.
#
# Submit with:  sbatch parameter_sweep.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=param-sweep
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --output=param-sweep_%j.out
#SBATCH --error=param-sweep_%j.err

echo "Parameter Sweep -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

# We'll compute the sum of 1..N using a Python loop (deliberately slow)
# for different values of N and measure how long each takes.

for N in 1000000 5000000 10000000 50000000; do

    echo "--- N = $N ---"

    # TODO: Call the sweep_compute.py script with $N as an argument.
    #
    # HINT: python3 sweep_compute.py $N
    #

    echo ""
done

echo "Sweep complete."

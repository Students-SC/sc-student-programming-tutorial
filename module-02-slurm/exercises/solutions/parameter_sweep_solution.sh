#!/bin/bash
# ---------------------------------------------------------------------------
# parameter_sweep_solution.sh -- Solution for the parameter sweep challenge
#
# Submit with:  sbatch solutions/parameter_sweep_solution.sh
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

for N in 1000000 5000000 10000000 50000000; do

    echo "--- N = $N ---"

    python3 solutions/sweep_compute_solution.py $N

    echo ""
done

echo "Sweep complete."

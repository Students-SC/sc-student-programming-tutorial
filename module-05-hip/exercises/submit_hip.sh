#!/bin/bash
# ---------------------------------------------------------------------------
# submit_hip.sh -- Run HIP exercises on a GPU node
#
# First compile:  hipcc -O2 -o vector_scale vector_scale.cpp
# Then submit:    sbatch submit_hip.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=hip-exercise
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --output=hip-exercise_%j.out
#SBATCH --error=hip-exercise_%j.err

echo "HIP GPU Exercise -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

echo "--- GPU Info ---"
rocm-smi --showproductname 2>/dev/null | head -15
echo ""

echo "--- Running vector_scale ---"
./vector_scale
echo ""

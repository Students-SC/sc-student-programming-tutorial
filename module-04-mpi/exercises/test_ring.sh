#!/bin/bash
# ---------------------------------------------------------------------------
# test_ring.sh -- Test a ring program with several rank counts in one job
#
# Each rank count runs as a step of this single job, so four tests use one
# compute-node allocation instead of four. timeout stops a run that is still
# waiting after 10 seconds (exit code 124, a deadlock), and the script goes on
# to the next count. The job ends FAILED if any count failed.
#
# Compile the program first, then submit with its name, for example:
#   sbatch test_ring.sh ./ring
#   sbatch test_ring.sh ./ring_ssend
# ---------------------------------------------------------------------------
#SBATCH --job-name=ring-test
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=2:00
#SBATCH --output=ring-test_%j.out

set -euo pipefail

PROGRAM=${1:?Usage: sbatch test_ring.sh ./ring_program}
FAILED=0

echo "Ring tests of $PROGRAM -- Job $SLURM_JOB_ID on $(hostname)"
for RANKS in 1 2 3 8; do
    echo ""
    echo "=== $RANKS rank(s) ==="
    # Record a failed count and go on to the next one.
    CODE=0
    srun --ntasks=$RANKS timeout 10 "$PROGRAM" || CODE=$?
    if [ "$CODE" -eq 124 ]; then
        echo "--> $RANKS rank(s): still waiting after 10 seconds, stopped (exit code 124)"
        FAILED=1
    elif [ "$CODE" -ne 0 ]; then
        echo "--> $RANKS rank(s): failed (exit code $CODE)"
        FAILED=1
    fi
done
exit "$FAILED"

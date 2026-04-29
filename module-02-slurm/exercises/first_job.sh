#!/bin/bash
# ---------------------------------------------------------------------------
# first_job.sh -- Your first Slurm batch job
#
# Submit with:  sbatch first_job.sh
#
# This job prints system info and then sleeps for 60 seconds, giving you
# time to practice squeue, scontrol show job, and scancel while it runs.
# ---------------------------------------------------------------------------
#SBATCH --job-name=first-job
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00
#SBATCH --output=first-job_%j.out
#SBATCH --error=first-job_%j.err

echo "========================================="
echo "  Hello from your first Slurm job!"
echo "========================================="
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Job name:      $SLURM_JOB_NAME"
echo "Hostname:      $(hostname)"
echo "Date:          $(date)"
echo "Partition:     $SLURM_JOB_PARTITION"
echo "Node list:     $SLURM_JOB_NODELIST"
echo "Tasks:         $SLURM_NTASKS"
echo "CPUs on node:  $SLURM_CPUS_ON_NODE"
echo "Working dir:   $(pwd)"
echo ""
echo "Now sleeping for 60 seconds..."

sleep 60

echo "Done sleeping. Job complete at $(date)."

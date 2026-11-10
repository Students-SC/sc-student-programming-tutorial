#!/bin/bash
#SBATCH --job-name=agent-job
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00
#SBATCH --output=agent-job_%j.out

./hello_compute

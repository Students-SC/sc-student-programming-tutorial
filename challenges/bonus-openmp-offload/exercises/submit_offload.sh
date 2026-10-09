#!/bin/bash
# ---------------------------------------------------------------------------
# submit_offload.sh -- Run the matmul_offload program on a GPU
#
# First compile (from a login node):
#   module load rocm openblas
#   amdclang -O2 -fopenmp --offload-arch=gfx90a -o matmul_offload matmul_offload.c \
#     -I$OPENBLAS_DIR/include -L$OPENBLAS_DIR/lib -lopenblas -lm
#
# Then submit:  sbatch submit_offload.sh
# ---------------------------------------------------------------------------
#SBATCH --job-name=omp-offload
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=5:00
#SBATCH --output=omp-offload_%j.out
#SBATCH --error=omp-offload_%j.err

module load rocm openblas

echo "OpenMP GPU Offload -- Job $SLURM_JOB_ID on $(hostname)"
echo ""

./matmul_offload

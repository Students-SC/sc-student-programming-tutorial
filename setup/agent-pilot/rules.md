# Rules for this cluster

- Frontier uses partition batch, account TRN001, and reservation
  sc26studentprogram. Each node has 64 CPU cores, 512 GB DRAM, and 8 MI250X
  GPU devices (64 GB HBM2e per device). Exercises may use only 16 CPU cores.
- AMD GPUs: use rocm-smi and rocminfo, never nvidia-smi. Compile GPU code
  with hipcc.
- Never run computations on the login node. Use srun or sbatch.
- Every srun and sbatch needs --partition and --time. Don't use --gpus.
- Compilers: gcc (OpenMP with -fopenmp), mpicc (Cray MPICH), hipcc (ROCm).

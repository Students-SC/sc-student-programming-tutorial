# Rules for this cluster

- Our partition is mi2101x: 16 CPU cores, 64 GB of memory, and 1 AMD MI210
  GPU per node.
- AMD GPUs: use rocm-smi and rocminfo, never nvidia-smi. Compile GPU code
  with hipcc.
- Never run computations on the login node. Use srun or sbatch.
- Every srun and sbatch needs --partition and --time. Don't use --gpus.
- Compilers: gcc (OpenMP with -fopenmp), mpicc (OpenMPI), hipcc (ROCm).

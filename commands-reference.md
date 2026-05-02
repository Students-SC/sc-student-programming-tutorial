# Commands Reference

Use this page when you remember the task but not the exact command. Commands are
grouped by workflow and link back to the module where they first appear.

## Getting Around

| I want to... | Command |
|--------------|---------|
| Show the current machine name | `hostname` |
| Show my username | `whoami` |
| Show my current directory | `pwd` |
| Change into a module exercise directory | `cd module-01-hpc-foundations/exercises` |
| List files | `ls` |
| List newest output files first | `ls -lt *.out \| head` |
| Print a file | `cat <file>` |
| Watch a job output file update | `tail -f <file>.out` |

## Cluster And Filesystem

| I want to... | Command |
|--------------|---------|
| Show home directory path | `echo $HOME` |
| Show work directory path | `echo $WORK` |
| Check home directory disk space | `df -h $HOME` |
| Check work directory disk space | `df -h $WORK` |
| Show memory usage | `free -h` |
| Show CPU details | `lscpu` |
| Show just the CPU model | `lscpu \| grep "Model name"` |

See [Module 1](module-01-hpc-foundations/README.md) for the first cluster
walkthrough.

## Software Modules

| I want to... | Command |
|--------------|---------|
| List loaded modules | `module list` |
| List available modules | `module avail` |
| Inspect the tutorial base module | `module show hpcfund` |

```{note}
The tutorial environment is preconfigured. Do not run `module purge` unless an
instructor tells you to.
```

## Slurm Jobs

| I want to... | Command |
|--------------|---------|
| Submit a batch job | `sbatch script.sh` |
| See my jobs | `squeue -u $USER` |
| See all jobs on the tutorial partition | `squeue -p mi2101x` |
| See partition and node status | `sinfo -p mi2101x` |
| See detailed node status | `sinfo -p mi2101x -N -l` |
| Get details for one job | `scontrol show job <JOBID>` |
| See accounting info after a job finishes | `sacct -j <JOBID>` |
| Cancel one job | `scancel <JOBID>` |
| Cancel all my jobs | `scancel -u $USER` |

See [Module 2](module-02-slurm/README.md) for the Slurm walkthrough.

## Quick Compute-Node Commands

| I want to... | Command |
|--------------|---------|
| Run `hostname` on a compute node | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 hostname` |
| Check CPU model on a compute node | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 lscpu \| grep "Model name"` |
| Preview ROCm GPU info on a compute node | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 rocminfo \| head -30` |
| Run a program on one compute node | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./program` |
| Run with 4 MPI ranks | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./program` |
| Run with 16 CPU cores for one task | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=16 ./program` |

## Slurm Script Directives

| Directive | Meaning |
|-----------|---------|
| `#SBATCH --job-name=name` | Name shown in the Slurm queue |
| `#SBATCH --partition=mi2101x` | Use the tutorial partition |
| `#SBATCH --nodes=1` | Request one node |
| `#SBATCH --ntasks=1` | Request one task or MPI rank |
| `#SBATCH --cpus-per-task=16` | Request CPU cores for one task |
| `#SBATCH --time=10:00` | Set the job time limit |
| `#SBATCH --output=name_%j.out` | Write stdout to a file; `%j` becomes the job ID |
| `#SBATCH --error=name_%j.err` | Write stderr to a file |

## Python Environment

| I want to... | Command |
|--------------|---------|
| Create the tutorial virtual environment from the repo root | `sbatch setup/setup_venv.sh` |
| Create the venv from inside a module exercise directory | `sbatch ../../setup/setup_venv.sh` |
| Activate the venv | `source "$WORK/sc26_venv/bin/activate"` |
| Verify PyTorch | `python3 -c "import torch; print(torch.__version__)"` |
| Verify ROCm is visible to PyTorch | `python3 -c "import torch; print(torch.cuda.is_available())"` |
| Leave the venv | `deactivate` |

## C And OpenMP

| I want to... | Command |
|--------------|---------|
| Compile a plain C program | `gcc -O2 -o program program.c` |
| Compile a C program that uses math functions | `gcc -O2 -o program program.c -lm` |
| Compile an OpenMP program | `gcc -fopenmp -O2 -o program program.c -lm` |
| Set the OpenMP thread count | `export OMP_NUM_THREADS=4` |
| Run OpenMP with 4 threads through Slurm | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=16 bash -c 'export OMP_NUM_THREADS=4; ./program'` |

Useful OpenMP snippets from [Module 3](module-03-openmp/README.md):

| OpenMP construct | Purpose |
|------------------|---------|
| `#pragma omp parallel` | Start a parallel region |
| `#pragma omp for` | Split loop iterations across threads |
| `#pragma omp parallel for` | Start threads and split a loop in one directive |
| `reduction(+:sum)` | Safely combine per-thread values into `sum` |
| `omp_get_thread_num()` | Get the current thread ID |
| `omp_get_num_threads()` | Get the size of the current thread team |

## MPI

| I want to... | Command |
|--------------|---------|
| Compile an MPI C program | `mpicc -O2 -o program program.c` |
| Compile an MPI C program using math functions | `mpicc -O2 -o program program.c -lm` |
| Run with 4 ranks | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./program` |
| Run with 16 ranks | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=16 ./program` |

Useful MPI calls from [Module 4](module-04-mpi/README.md):

| MPI call | Purpose |
|----------|---------|
| `MPI_Init(&argc, &argv)` | Initialize MPI |
| `MPI_Finalize()` | Shut down MPI |
| `MPI_Comm_rank(comm, &rank)` | Get this rank's ID |
| `MPI_Comm_size(comm, &size)` | Get the number of ranks |
| `MPI_Send(...)` | Send data to another rank |
| `MPI_Recv(...)` | Receive data from another rank |
| `MPI_Reduce(...)` | Combine values across ranks |
| `MPI_Bcast(...)` | Broadcast a value from one rank to all ranks |

## HIP And GPU Tools

| I want to... | Command |
|--------------|---------|
| Check whether `hipcc` is available | `which hipcc` |
| Show the HIP compiler version | `hipcc --version` |
| Show ROCm system info | `rocminfo \| head -20` |
| Show GPU memory usage | `rocm-smi --showmeminfo vram` |
| Compile a HIP program | `hipcc -O2 -o program program.cpp` |
| Compile with a block-size macro | `hipcc -O2 -DBLOCK_SIZE=256 -o program program.cpp` |
| Run a HIP program on a compute node | `srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./program` |

Useful HIP calls and syntax from [Module 5](module-05-hip/README.md):

| HIP item | Purpose |
|----------|---------|
| `hipMalloc(&ptr, bytes)` | Allocate memory on the GPU |
| `hipMemcpy(dst, src, bytes, dir)` | Copy data between host and GPU |
| `hipFree(ptr)` | Free GPU memory |
| `hipDeviceSynchronize()` | Wait for GPU work to finish |
| `hipGetLastError()` | Check for kernel launch errors |
| `__global__ void kernel(...)` | Declare a GPU kernel |
| `kernel<<<blocks, threads>>>(...)` | Launch a kernel |
| `threadIdx.x` | Thread ID within a block |
| `blockIdx.x` | Block ID within the grid |
| `blockDim.x` | Threads per block |

## AI Exercises

| I want to... | Command |
|--------------|---------|
| Run the inference batch script | `sbatch submit_inference.sh` |
| Read inference output | `cat inference_<JOBID>.out` |
| Run the fine-tuning batch script | `sbatch submit_finetune.sh` |
| Read fine-tuning output | `cat finetune_<JOBID>.out` |
| Load a text generation model in Python | `AutoModelForCausalLM.from_pretrained(name)` |
| Load a tokenizer in Python | `AutoTokenizer.from_pretrained(name)` |
| Move a PyTorch model to the GPU | `model.to("cuda")` |
| Generate text with a model | `model.generate(input_ids, ...)` |

See [Module 6](module-06-ai-inference-finetuning/README.md) for inference and
fine-tuning details.

## AI Agent

| I want to... | Command |
|--------------|---------|
| Show the shared agent server URL | `cat "$WORK/sc26_agent_server_url"` |
| Export the agent server URL for interactive testing | `export AGENT_API_URL=$(cat "$WORK/sc26_agent_server_url")` |
| Run the agent exercise | `sbatch submit_agent.sh` |
| Launch the tutorial-provided CLI agent | `bash ../../setup/launch_aider.sh` |
| Launch aider from an arbitrary working directory | `bash <repo-path>/setup/launch_aider.sh` |

See [Module 7](module-07-ai-agents/README.md) for the agent exercises.

## Common Output Files

| Job or exercise | Output file pattern |
|-----------------|---------------------|
| Venv setup | `setup_venv_<JOBID>.out` |
| Slurm first job | `first-job_<JOBID>.out` |
| Compute-node hello | `hello-compute_<JOBID>.out` |
| OpenMP pi | `openmp-pi_<JOBID>.out` |
| MPI sum | `mpi-sum_<JOBID>.out` |
| HIP exercise | `hip-exercise_<JOBID>.out` |
| Inference | `inference_<JOBID>.out` |
| Fine-tuning | `finetune_<JOBID>.out` |
| Agent exercise | `agent_<JOBID>.out` |

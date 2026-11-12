# Challenges

This section is the home for optional challenges that extend the tutorial
modules.

Challenge activities will be added here as module owners review their sections
of the curriculum.

## Module 3 - OpenMP Challenges

Starting from the tutorial repository's root directory, navigate to the Module
3 challenges directory:

```bash
cd challenges/module-03-openmp-challenges
```



## Module 4: MPI Challenges

### Hybrid MPI + OpenMP

Large applications often combine both models: a few MPI ranks per node, each
running several OpenMP threads. In this challenge, you turn Module 4's
parallel sum into a hybrid program and compare different ways of splitting a
node's 16 cores between ranks and threads.

#### Step 1: Open the Template

Starting from the tutorial repository's root directory, navigate to the MPI
challenges directory and open the template in an editor:

```bash
cd challenges/module-04-mpi-challenges
vim hybrid_sum.c
```

#### Step 2: Complete the TODOs

There are **2 TODOs** to fill in:

1. **TODO 1**: Initialize MPI with `MPI_Init_thread`, requesting
   `MPI_THREAD_FUNNELED` (threads may run, but only the main thread makes MPI
   calls), and check the level that MPI provides.
2. **TODO 2**: Split each rank's loop across OpenMP threads, combining the
   threads' results with a reduction, as in Module 3.

#### Step 3: Compile and Submit

```bash
mpicc -fopenmp -O2 -o hybrid_sum hybrid_sum.c -lm
sbatch submit_hybrid.sh
```

The batch script runs four splits of the node's 16 cores: 16 ranks x 1
thread, 4 x 4, 2 x 8, and 1 x 16. For each one, `srun --cpus-per-task` gives
every rank its own cores for its threads.

#### Step 4: Examine the Output

```bash
cat hybrid-sum_<JOBID>.out
```

**Questions:**
- Do all four splits give the same sum? About the same time?
- If the time grows as the number of ranks drops, what does that tell you
  about your threads?
- On one node, the splits can perform about the same. On a large cluster, why
  might an application prefer fewer ranks with more threads on each node?
  (Hint: think about how many messages are exchanged, and how much memory
  each rank uses.)

A completed version is in `solutions/hybrid_sum_solution.c`.

# Bonus Modules

## OpenMP Offload

This bonus module covers OpenMP directives for GPU programming. This requires completing [Module 3 OpenMP](../module-03-openmp/README.md) and [Module 5 HIP](../module-05-hip/README.md). [Click here to get started](./bonus-openmp-offload/README).


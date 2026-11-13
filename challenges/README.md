# Challenges

This section is the home for optional challenges that extend the tutorial
modules.

Challenge activities will be added here as module owners review their sections
of the curriculum.



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

Note the job ID printed by `sbatch`. Wait for the job to finish, using
`squeue -u $USER` to check pending and running jobs. Disappearing from that
list does not establish success.

#### Step 4: Examine the Output

Replace `<JOBID>` with the ID of your submission:

```bash
cat hybrid-sum_<JOBID>.out
cat hybrid-sum_<JOBID>.err
sacct -j <JOBID> --format=JobID,JobName%20,State,ExitCode
```

Check that the job and all four numbered run steps are `COMPLETED` with exit
code `0:0`, and inspect the error log. If accounting is initially empty, try
again shortly. The script stops at the first failed run; do not interpret an
incomplete sweep as a performance comparison.

Expect a sum of approximately `2.11e13`, consistent with Module 4's one-rank
baseline. Small differences in the last digits can result from changing the
addition order. The reported time is rank 0's calculation and reduction
interval, not job startup or a repeated benchmark. The threads label reports
`omp_get_max_threads()`, the configured maximum; it does not prove that the
loop actually used a team of threads.

**Questions:**
- Do all four splits give the same sum? About the same time?
- If the time grows as the number of ranks drops, what does that tell you
  about your threads?
- All four splits use the same node's 16 cores. How does the amount of private
  process memory and the number of MPI contributions change as you use fewer
  ranks and more threads? How do threads combine their partial sums before
  MPI combines the ranks' results?
- On one node, the splits can perform about the same. On a large cluster, why
  might an application prefer fewer ranks with more threads on each node?
  (Hint: think about how many messages are exchanged, and how much memory
  each rank uses.)

A completed version is in `solutions/hybrid_sum_solution.c`.

# Bonus Modules

## OpenMP Offload

This bonus module covers OpenMP directives for GPU programming. This requires completing [Module 3 OpenMP](../module-03-openmp/README.md) and [Module 5 HIP](../module-05-hip/README.md). [Click here to get started](./bonus-openmp-offload/README).


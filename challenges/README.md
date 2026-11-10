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

### Challenge A: Matrix-Vector Multiply

Parallelize a matrix-vector multiplication using OpenMP. A template is at:

```bash
cat matvec_openmp.c
```

The outer loop over rows is embarrassingly parallel -- each row of the output
can be computed independently. Add the appropriate OpenMP directive and compare
performance with the serial version. Compile and run it with:

```bash
gcc -fopenmp -O2 -o matvec_openmp matvec_openmp.c -lm
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=16 \
  bash -c 'export OMP_NUM_THREADS=16; ./matvec_openmp'
```

**Question:** On 16 threads, the matrix-vector multiply speeds up much less
than the pi program did in Module 3's Exercise 3. Why? (Hint: compare how much
data each program reads from memory for each arithmetic operation it does.)

A completed version is in
[`module-03-openmp/exercises/solutions/matvec_openmp_solution.c`](../module-03-openmp/exercises/solutions/matvec_openmp_solution.c).



### Challenge B: Exploring other clauses for `pragma omp for`

If you look at the [OpenMP API spec for `pragma for`](https://www.openmp.org/spec-html/5.0/openmpsu41.html#x64-1290002.9.2), you will see it has a number of optional clauses you can use. You've seen one already: `reduction`. Try exploring some of the others. For example:

1. Could you apply the `collapse` clause to the loops in `matvec_openmp.c`? Does it compile? If not, read the compiler's error message: what does it say about the loops, and why?
2. Could you apply the `schedule` clause to the loop in your completed `pi_openmp.c` (in `module-03-openmp/exercises/`) and to `matvec_openmp.c`? Is there any difference in execution time between `schedule(static)` and `schedule(dynamic)`? If `dynamic` is much slower, try giving it a chunk size, such as `schedule(dynamic, 100000)`. Why does that help?


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


# Challenges

This section is the home for optional challenges that extend the tutorial
modules.

Challenge activities will be added here as module owners review their sections
of the curriculum.

## Module 3 - OpenMP Challenges

The files for these challenges are located in `module-03-openmp-challenges` directory.

### Challenge A: Find the Race Condition

The file `pi_race.c` has a **deliberate bug** -- a
race condition. Compile and run it:

```bash
gcc -fopenmp -O2 -o pi_race pi_race.c -lm
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=16 \
  bash -c 'export OMP_NUM_THREADS=8; ./pi_race'
```

Run it several times. Notice the answer changes each time! Can you spot and fix
the bug? (Hint: compare it to your working `pi_openmp.c`.)


### Challenge B: Matrix-Vector Multiply

Parallelize a matrix-vector multiplication using OpenMP. A template is at:

```bash
cat matvec_openmp.c
```

The outer loop over rows is embarrassingly parallel -- each row of the output
can be computed independently. Add the appropriate OpenMP directive and compare
performance with the serial version.



### Challenge C: Exploring other clauses for `pragma omp for`

If you look at the [OpenMP API spec for `pragma for`](https://www.openmp.org/spec-html/5.0/openmpsu41.html#x64-1290002.9.2), you will see it has a number of optional clauses you can use. You've seen one already: `reduction`. Try exploring some of the others. For example:

1. Could you apply the `collapse` clause for the `matvec_openmp.c` exercise? Does that still produce the correct output? If not, why?
2. Could you apply the `schedule` clause for the `pi_race.c` and for `matvec_openmp.c`? Is there any difference in execution time between using `schedule(static)` and `schedule(dynamic)` for `omp parallel for`?


## Module 4: MPI Challenges

### Ring Communication

Run all ranks on one compute node. In a ring pattern, each rank sends its
rank number to the next rank and receives the previous rank's number. Each
value moves one hop. The last rank sends back to rank 0:

```text
  Rank 0 → Rank 1 → Rank 2 → Rank 3
    ▲                         │
    └─────────────────────────┘
```

#### Step 1: Open the Template

Starting from the tutorial repository's root directory, navigate to the MPI
challenges directory and open the template in an editor:

```bash
cd challenges/module-04-mpi-challenges
vim ring.c
```

In Vim, press `i` to edit. When you finish, press `Esc`, type `:wq`, and press
`Enter` to save and exit.

#### Step 2: Complete the Communication

Fill in the TODO to send `send_val` to `dest` and receive a value from
`source` into `recv_val`. Use `MPI_Send` and `MPI_Recv`, or explore
`MPI_Sendrecv` as an alternative.

```{tip}
Plan the order of the sends and receives. If every rank starts with a blocking
receive, no rank can send. Starting with blocking sends can also deadlock if
the messages cannot be buffered. Consider alternating the order for even and
odd ranks, or use `MPI_Sendrecv` to coordinate the exchange.
```

#### Step 3: Compile and Run

Compile on the login node:

```bash
mpicc -O2 -o ring ring.c
```

Run on a compute node with four ranks:

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./ring
```

Each rank should receive the previous rank's number. With four ranks, rank 0
should receive `3` from rank 3 and report that ring communication succeeded.
The output lines may appear in any order.

Check every rank's source, destination, and received value. The success banner
checks only rank 0; the program exits with status 0 even when it prints a
failed value check.

Try again with eight ranks by changing `--ntasks=4` to `--ntasks=8`. Rank 0
should now receive `7`. In general, rank 0 receives `size - 1`.

#### Step 4: Compare with the Solution

After completing your implementation, examine the solution in
`solutions/ring_solution.c`. It uses `MPI_Sendrecv` to exchange values.
From the same challenges directory, compile and run it separately:

```bash
mpicc -O2 -o ring_solution solutions/ring_solution.c
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./ring_solution
```

# Bonus Modules

## OpenMP Offload

This bonus module covers OpenMP directives for GPU programming. This requires completing [Module 3 OpenMP](../module-03-openmp/README.md) and [Module 5 HIP](../module-05-hip/README.md). [Click here to get started](./bonus-openmp-offload/README).


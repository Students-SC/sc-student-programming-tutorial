# Module 4 -- Distributed-Memory Parallelism with MPI

> **Time:** 11:10 AM to 12:00 PM CST · 50 min total · ~15 min lecture · ~35 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain the message-passing model and how it differs from shared memory
- Write an MPI program that distributes work across multiple processes (ranks)
- Explain point-to-point communication (`MPI_Send` / `MPI_Recv`)
- Use collective operations (`MPI_Reduce`)

---

## Key Concepts

### Why MPI?

OpenMP threads share memory within a single node, but what if you need more
compute power than one node provides? **MPI (Message Passing Interface)**
lets you run a program across **multiple processes** -- on the same node or
across many nodes -- that communicate by **sending and receiving messages**.

```
    Node 0                  Node 1
  ┌──────────┐           ┌──────────┐
  │ Rank 0   │◄─────────►│ Rank 2   │
  │ Rank 1   │  messages  │ Rank 3   │
  └──────────┘  (network) └──────────┘
```

Each process is called a **rank** and has its own private memory. Ranks
coordinate by explicitly sending and receiving data.

### MPI Basics

Every MPI program follows this structure:

```c
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);          // Start MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // My rank (0, 1, 2, ...)
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Total number of ranks

    // ... do work ...

    MPI_Finalize();                  // Shut down MPI
    return 0;
}
```

**Key rule:** Every rank executes the **same program**, but branches based on
its rank number to do different work.

### Point-to-Point Communication

Send a message from one rank to another. This example requires at least two ranks:

```c
if (rank == 0) {
    int data = 42;
    MPI_Send(&data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);  // Send to rank 1
} else if (rank == 1) {
    int data;
    MPI_Recv(&data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Rank 1 received: %d\n", data);  // Prints 42
}
```

### Collective Operations

Instead of manually sending between every pair of ranks, MPI provides
**collectives** that coordinate all ranks at once:

| Operation | What it does |
|-----------|-------------|
| `MPI_Bcast` | One rank sends the same data to all others |
| `MPI_Reduce` | All ranks contribute a value; one rank gets the combined result |
| `MPI_Allreduce` | Like Reduce, but every rank gets the result |
| `MPI_Scatter` | One rank distributes different pieces to each rank |
| `MPI_Gather` | Each rank sends a piece; one rank collects them all |

Example -- sum a value across all ranks:

```c
double local_sum = compute_my_part(rank);
double global_sum;
MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
if (rank == 0)
    printf("Global sum = %f\n", global_sum);
```

### Compiling and Running

Compile these small programs on the login node with `mpicc`, then use `srun`
to execute them on a compute node:

```bash
mpicc -O2 -o program program.c
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./program
```

On our cluster, `srun` automatically launches one copy of your program per
rank, configured via `--ntasks`.

---

## Hands-On Exercises (~35 min)

Suggested pacing: about 8 minutes for Hello World, 20 minutes for Parallel Sum,
and 7 minutes for interactive MPI. Complete the main exercises before trying
the optional challenges.

From the repository root on the login node, navigate to this module's exercises:

```bash
cd module-04-mpi/exercises
```

### Exercise 1: Run an MPI Hello World

#### Step 1: Examine the Program

Examine the MPI hello-world program:

```bash
cat ../examples/mpi_hello.c
```

#### Step 2: Compile on the Login Node

```bash
mpicc -O2 -o mpi_hello ../examples/mpi_hello.c
```

#### Step 3: Run with Four Ranks

Run the program on a compute node with four ranks:

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./mpi_hello
```

#### Step 4: Experiment with Rank Counts

Repeat the command with `--ntasks=1` and compare it with the four-rank output.
Each rank reports a different ID, but the ranks may print in any order (that's
normal for parallel programs). If time permits, also try 8 and 16 ranks.

---

### Exercise 2: Parallel Sum with MPI_Reduce

#### Step 1: Open the Template

Open the exercise template in an editor on the login node:

```bash
vim parallel_sum.c
```

In Vim, press `i` to enter insert mode and make your edits. Press `Esc`, type
`:wq`, and press `Enter` to save and exit. You can instead use `nano`, `emacs`,
or an existing VS Code Remote SSH connection.

The program divides a large array across ranks: each rank computes the sum of
its portion, then `MPI_Reduce` combines the partial sums into a global total.

#### Step 2: Complete the TODOs

There are **4 TODOs** to fill in:

1. **TODO 1**: Initialize MPI
2. **TODO 2**: Get the rank and size
3. **TODO 3**: Use `MPI_Reduce` to sum the partial results
4. **TODO 4**: Finalize MPI

#### Step 3: Compile on the Login Node

After filling in the TODOs, compile the program:

```bash
mpicc -O2 -o parallel_sum parallel_sum.c -lm
```

#### Step 4: Submit the Batch Job

The batch script runs your completed program with 1, 2, 4, 8, and 16 ranks on
one compute node:

```bash
sbatch submit_mpi.sh
```

Note the job ID printed by `sbatch` and wait for the job to finish. You can
check its status with `squeue -u $USER`.

#### Step 5: Examine the Output

Replace `<JOBID>` with your job ID:

```bash
cat mpi-sum_<JOBID>.out
```

Use the one-rank global sum as the baseline for comparing the results at other
rank counts. They should agree closely, although floating-point addition can
produce small differences in the last digits when the order of addition changes.

**Questions:**
- Do the global sums at 2, 4, 8, and 16 ranks agree with the one-rank baseline?
- How does the time change from 1 rank to 16 ranks?
- What would happen if you forgot the `MPI_Reduce` and just printed each
  rank's partial sum?

---

### Exercise 3: Run MPI Interactively

#### Step 1: Compile the Program

Use the completed `parallel_sum.c` from Exercise 2. If you have changed it,
recompile it on the login node:

```bash
mpicc -O2 -o parallel_sum parallel_sum.c -lm
```

#### Step 2: Try Different Rank Counts

Run the program directly with several rank counts:

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=2  ./parallel_sum
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=8  ./parallel_sum
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=16 ./parallel_sum
```

---

### Additional Challenges

After completing the main modules, try the
[MPI Ring Communication challenge](../challenges/README.md#module-4-mpi-challenges)
to practice point-to-point communication and reason about deadlock.

---

## Quick Reference

| Function | Purpose |
|----------|---------|
| `MPI_Init(&argc, &argv)` | Initialize MPI |
| `MPI_Finalize()` | Shut down MPI |
| `MPI_Comm_rank(comm, &rank)` | Get my rank ID |
| `MPI_Comm_size(comm, &size)` | Get total number of ranks |
| `MPI_Send(buf, count, type, dest, tag, comm)` | Send data to a rank |
| `MPI_Recv(buf, count, type, src, tag, comm, status)` | Receive data from a rank |
| `MPI_Reduce(sendbuf, recvbuf, count, type, op, root, comm)` | Combine values across ranks |
| `MPI_Bcast(buf, count, type, root, comm)` | Broadcast from one rank to all |

| MPI Datatype | C Type |
|-------------|--------|
| `MPI_INT` | `int` |
| `MPI_LONG` | `long` |
| `MPI_DOUBLE` | `double` |
| `MPI_CHAR` | `char` |

| Reduction Op | Meaning |
|-------------|---------|
| `MPI_SUM` | Sum |
| `MPI_MAX` | Maximum |
| `MPI_MIN` | Minimum |

---

**Next up:** [Module 5 -- GPU Programming with HIP](../module-05-hip/README.md)

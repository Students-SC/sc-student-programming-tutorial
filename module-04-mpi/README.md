# Module 4 -- Distributed-Memory Parallelism with MPI

> **Time:** 11:10 AM to 12:00 PM CST · 50 min total · ~15 min lecture · ~35 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain the message-passing model and how it differs from shared memory
- Write an MPI program that distributes work across multiple processes (ranks)
- Use point-to-point communication (`MPI_Send` / `MPI_Recv`)
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

Send a message from one rank to another:

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

```bash
mpicc -O2 -o program program.c      # Compile with MPI
srun --ntasks=4 ./program            # Run with 4 ranks
```

On our cluster, `srun` automatically launches one copy of your program per
rank, configured via `--ntasks`.

---

## Hands-On Exercises (~35 min)

First, navigate to the exercises directory for this module:

```bash
cd module-04-mpi/exercises
```

### Step 0: Look at the Example

Examine the MPI hello-world program:

```bash
cat ../examples/mpi_hello.c
```

Compile and run it on a compute node with 4 ranks:

```bash
mpicc -O2 -o mpi_hello ../examples/mpi_hello.c
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./mpi_hello
```

Try running with 1, 4, 8, and 16 ranks. Notice that each rank reports a
different ID, but they may print in any order (that's normal for parallel
programs).

---

### Exercise 1: Parallel Sum with MPI_Reduce (Core)

Open the exercise template:

```bash
cat parallel_sum.c
```

The program divides a large array across ranks: each rank computes the sum of
its portion, then `MPI_Reduce` combines the partial sums into a global total.

There are **4 TODOs** to fill in:

1. **TODO 1**: Initialize MPI
2. **TODO 2**: Get the rank and size
3. **TODO 3**: Use `MPI_Reduce` to sum the partial results
4. **TODO 4**: Finalize MPI

After filling in the TODOs, compile and run:

```bash
mpicc -O2 -o parallel_sum parallel_sum.c -lm
```

Submit the batch script:

```bash
sbatch submit_mpi.sh
```

Check the output:

```bash
cat mpi-sum_<JOBID>.out
```

**Questions:**
- Does the parallel result match the serial reference?
- How does the time change from 1 rank to 16 ranks?
- What would happen if you forgot the `MPI_Reduce` and just printed each
  rank's partial sum?

---

### Exercise 2: Interactive MPI (Core)

Try running with different rank counts directly:

```bash
mpicc -O2 -o parallel_sum parallel_sum.c -lm
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=2  ./parallel_sum
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=8  ./parallel_sum
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=16 ./parallel_sum
```

---

### Challenge A: Ring Communication

In a ring pattern, each rank sends data to the next rank and receives from the
previous rank, forming a circle:

```
  Rank 0 → Rank 1 → Rank 2 → Rank 3
    ▲                               │
    └───────────────────────────────┘
```

Open the template:

```bash
cat ring.c
```

Fill in the TODOs to implement the ring using `MPI_Send` and `MPI_Recv`. Each
rank sends its rank number to the next; after going around the ring, rank 0
should have received `size - 1` (the last rank's number).

```{tip}
Think carefully about who sends first to avoid **deadlock** (all ranks waiting
to receive before anyone sends).
```

### Challenge B: Ask the Agent

Describe this problem to your AI agent: *"Write an MPI program in C where each
rank generates a random number, and rank 0 prints the minimum, maximum, and
average across all ranks."* Review the generated code for:

- Does it call `MPI_Init` and `MPI_Finalize`?
- Does it use the right MPI datatypes?
- Could it deadlock?
- Does rank 0 print the correct results?

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

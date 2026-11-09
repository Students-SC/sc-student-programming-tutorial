# Module 4 -- Distributed-Memory Parallelism with MPI

> **Time:** 11:10 AM to 12:00 PM CST · 50 min total · ~15 min lecture · ~35 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain the message-passing model and how it differs from shared memory
- Write an MPI program that distributes work across multiple processes (ranks) on one node
- Explain point-to-point communication (`MPI_Send` / `MPI_Recv`)
- Recognize a deadlock and order messages to avoid it
- Use collective operations (`MPI_Reduce`)
- Test an agent's MPI code for a deadlock that a normal run can hide

---

## Key Concepts

### Why MPI?

As you saw in Module 3, OpenMP threads work together through **shared
memory**, so they must all run on one compute node: each node has its own
memory, and threads can't see another node's. **MPI (Message Passing
Interface)** takes a different approach. A program runs as **multiple
processes**, called **ranks**, and each rank has its own private memory.
Ranks coordinate by explicitly **sending and receiving messages**, and because
messages can travel over the network, an MPI program can run across many
compute nodes.

That is how programs grow beyond one node. The same program can run across
many nodes of a real cluster, for example as 64 ranks with
`srun --nodes=4 --ntasks-per-node=16 ./program`. Ranks on the same node
exchange messages through shared memory, while ranks on different nodes send
them over the high-speed network from Module 1.

In this tutorial, we focus on MPI within a single node: our tutorial partition
allows one node per job, so in this module you run 1 to 16 ranks on the CPU
cores of one compute node.

```
  One compute node
  ┌───────────────────────────────────────────────────┐
  │ ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐ │
  │ │ Rank 0 │◄─►│ Rank 1 │◄─►│ Rank 2 │◄─►│ Rank 3 │ │
  │ └────────┘   └────────┘   └────────┘   └────────┘ │
  │               messages between ranks              │
  └───────────────────────────────────────────────────┘
```

Single-node MPI is still useful. Each rank keeps its own private memory even
though all ranks share a node, so you practice exactly the programming model
that runs across nodes, and the programs you write today would run on many
nodes without changes.

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

MPI programs are compiled with `mpicc`, a wrapper around `gcc` that adds the
MPI headers and libraries, and run with `srun`, which starts one copy of the
program for each rank. In general, for a program in `program.c`:

```bash
mpicc -O2 -o program program.c                                       # Compile (fine on the login node)
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./program  # Run 4 ranks on a compute node
```

`--nodes=1` places all ranks on one compute node. `--ntasks=4` tells `srun`
to launch four ranks, each running one copy of the program. A node has 16 CPU
cores, so you can use up to 16 ranks. You'll compile and run real programs
this way in the exercises below.

---

## Hands-On Exercises (~35 min)

Suggested pacing: about 8 minutes for Hello World, 17 minutes for Parallel Sum,
and 10 minutes for the ring. Complete the main exercises before trying the
optional challenges.

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
normal for parallel programs). You'll run up to 16 ranks in Exercise 2's batch
job.

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

The program divides an index range across ranks. Each rank computes
`sqrt(i)` for the indices in its range and adds those values locally, then
`MPI_Reduce` combines the partial sums into a global total. No array is stored
or distributed.

#### Step 2: Complete the TODOs

There are **4 TODOs** to fill in:

1. **TODO 1**: Initialize MPI
2. **TODO 2**: Get the rank and size
3. **TODO 3**: Use `MPI_Reduce` to sum the partial results
4. **TODO 4**: Finalize MPI

Replace the placeholder rank/size setup instead of adding a second declaration.
Place the reduction after the declaration of `global_sum` and outside the
rank-0 printing condition so every rank participates.

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

With the default 1 billion terms, expect a positive sum of approximately
`2.11e13` (about 21,082,000,000,000). Check this magnitude first, then use the one-rank global sum as a
consistency baseline for other rank counts. They should agree closely,
although floating-point addition can produce small differences in the last
digits when the order of addition changes.

The reported time covers rank 0's local calculation and reduction interval;
it excludes job startup. Use it to explore scaling. Reliable benchmarking
would need repeated measurements and coordinated timing across ranks.

Copy this table into your notes and fill it in from the output, as you did
for OpenMP in Module 3. Speedup is the 1-rank time divided by the N-rank time:

| Ranks | Time (s) | Speedup |
|-------|----------|---------|
| 1     |          | 1.0     |
| 2     |          |         |
| 4     |          |         |
| 8     |          |         |
| 16    |          |         |

**Questions:**
- Does the sum have the expected magnitude?
- Do the global sums at 2, 4, 8, and 16 ranks agree with the one-rank baseline?
- How does the time change from 1 rank to 16 ranks? How does your speedup
  compare with your OpenMP pi speedup from Module 3? If both scale about as
  well on one node, why would you choose MPI? (Hint: see "Why MPI?" above.)
- What would happen if you forgot the `MPI_Reduce` and just printed each
  rank's partial sum?

#### Step 6: Check the Remainder Handling

Run your program once more, directly with `srun`, using three ranks:

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=3 ./parallel_sum
```

1 billion is not divisible by three, so the last rank computes one extra term.
Its global sum should still agree closely with the one-rank baseline.

---

### Exercise 3: Pass Messages Around a Ring

In a ring pattern, each rank sends its rank number to the next rank and
receives the previous rank's number. Each value moves one hop. The last rank
sends back to rank 0:

```text
  Rank 0 → Rank 1 → Rank 2 → Rank 3
    ▲                         │
    └─────────────────────────┘
```

#### Step 1: Open the Template

Open the template in an editor on the login node:

```bash
vim ring.c
```

#### Step 2: See a Deadlock First

Start with the simplest idea: every rank sends, then receives. Put these two
lines at the TODO:

```c
MPI_Ssend(&send_val, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
MPI_Recv(&recv_val, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
```

Compile and run it with a short time limit:

```bash
mpicc -O2 -o ring ring.c
srun --partition=mi2101x --nodes=1 --time=1:00 --ntasks=4 ./ring
```

The program hangs with no output. `MPI_Ssend` (synchronous send) does not
finish until the receiving rank has started its receive, but every rank is
stuck sending, so no rank ever reaches `MPI_Recv`. This is a **deadlock**.
Press `Ctrl-C` twice within a second to stop `srun` (otherwise the job holds a
whole node until its 1-minute limit).

With plain `MPI_Send`, this same code often works, because MPI copies small
messages into a buffer and returns right away. Larger messages, or another
MPI library, can make it deadlock, so correct programs never rely on that
buffering.

#### Step 3: Fix the Ordering

Replace those two lines so the exchange can't deadlock. Fill in the TODO to
send `send_val` to `dest` and receive a value from `source` into `recv_val`.
Use `MPI_Send` and `MPI_Recv`, or explore `MPI_Sendrecv` as an alternative. A
correct ordering works even with `MPI_Ssend`, which makes it a good test.

```{tip}
Plan the order of the sends and receives. If every rank starts with a blocking
receive, no rank can send. Starting with blocking sends can also deadlock if
the messages cannot be buffered. Consider alternating the order for even and
odd ranks, or use `MPI_Sendrecv` to coordinate the exchange.
```

#### Step 4: Compile and Run

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

#### Step 5: Compare with the Solution

After completing your implementation, examine the solution in
`solutions/ring_solution.c`. It uses `MPI_Sendrecv` to exchange values.
Compile and run it separately:

```bash
mpicc -O2 -o ring_solution solutions/ring_solution.c
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./ring_solution
```

---

### Agent Practice: Test the Agent's Ring

Code that runs once isn't necessarily correct. In Step 2 you saw a ring that
deadlocks when messages aren't buffered. Now have the agent write a ring, and
use the same trick to prove whether its version can deadlock.

#### Step 1: Ask for a Ring

Start Qwen Code:

```bash
bash ../../setup/launch_qwen.sh
```

Ask it:

> Write ring_agent.c: an MPI program in which every rank sends its rank number to the next rank (the last rank sends to rank 0) and receives from the previous rank, then prints a line 'Rank <r> received <v>'. Use MPI_Send and MPI_Recv.

Before you approve the file, read the sends and receives. Predict: would this
version deadlock if MPI couldn't buffer the messages?

#### Step 2: Test It Without Buffering

Make a copy that uses `MPI_Ssend`, then compile and run it with a short time
limit:

```bash
sed 's/MPI_Send(/MPI_Ssend(/g' ring_agent.c > ring_ssend.c
mpicc -O2 -o ring_ssend ring_ssend.c
srun --partition=mi2101x --nodes=1 --time=1:00 --ntasks=4 ./ring_ssend
```

If it hangs, the agent's ordering relies on buffering: press `Ctrl-C` twice,
tell the agent what happened, and ask it to fix the ordering. When it works,
run it again with `--ntasks=8` and check every rank's received value.

**Questions:**
- Which ordering did the agent use? Is it the same idea as yours?
- Did it follow the instruction to use `MPI_Send` and `MPI_Recv`? If not, did
  you notice before running it?
- Why does the `MPI_Ssend` test tell you more than a run that happens to work?

---

### Additional Challenges

If you finish the exercises above early, try the
[hybrid MPI + OpenMP challenge](../challenges/README.md#module-4-mpi-challenges):
combine MPI ranks with OpenMP threads and compare ways of splitting a node's
cores between them. You can also come back to it after the tutorial.

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
| `MPI_Ssend(buf, count, type, dest, tag, comm)` | Send, waiting until the receiver has started receiving |
| `MPI_Sendrecv(...)` | Send to one rank and receive from another in one call |
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

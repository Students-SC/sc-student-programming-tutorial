# Challenges

This section is the home for optional challenges that extend the tutorial
modules.

Challenge activities will be added here as module owners review their sections
of the curriculum.

## Module 4: MPI Challenges

### Ring Communication

In a ring pattern, each rank sends its rank number to the next rank and
receives the previous rank's number. Each value moves one hop. The last rank
sends back to rank 0:

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

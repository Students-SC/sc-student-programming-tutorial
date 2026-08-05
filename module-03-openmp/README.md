# Module 3 -- Shared-Memory Parallelism with OpenMP

> **Time:** 10:20 to 11:10 AM CST · 50 min total · ~15 min lecture · ~35 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Explain the difference between threads and processes
- Add OpenMP directives to parallelize a serial C program
- Control the number of threads and measure speedup
- Identify and fix race conditions in shared-memory code

---

## Key Concepts

### Threads vs. Processes

| | Processes | Threads |
|---|---|---|
| Memory | Separate (private) | Shared |
| Communication | Message passing (MPI) | Read/write shared variables |
| Creation cost | Higher | Lower |
| Best for | Multi-node | Single node |

**OpenMP** uses **threads** -- lightweight workers that all share the same memory
space within a single node. This makes it easy to parallelize loops: all threads
can see the same arrays and variables.

### The OpenMP Model

OpenMP uses **compiler directives** (special comments that the compiler
understands) to mark regions of code for parallel execution.

```
          Serial            Parallel (fork)           Serial (join)
         ─────────►     ┌── Thread 0 ──┐           ─────────►
                        ├── Thread 1 ──┤
                        ├── Thread 2 ──┤
                        └── Thread 3 ──┘
```

This is called **fork-join** parallelism: the program forks into multiple
threads, they do work in parallel, then they join back into one.

### Essential Directives

**Make a region parallel** -- every thread executes the block:

```c
#pragma omp parallel
{
    printf("Hello from thread %d\n", omp_get_thread_num());
}
```

**Distribute loop iterations** across threads:

```c
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    a[i] = b[i] + c[i];
}
```

**Reduction** -- safely combine a value across threads:

```c
double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++) {
    sum += a[i];
}
```

Without `reduction`, multiple threads writing to `sum` simultaneously would
cause a **race condition** -- a bug where the result depends on unpredictable
timing.

### Decomposing `parallel for`

The combined `parallel for` is really *two* directives fused together for
convenience:

```c
#pragma omp parallel for reduction(+:sum)
for (...) { ... }
```

is exactly equivalent to:

```c
#pragma omp parallel
{
    #pragma omp for reduction(+:sum)
    for (...) { ... }
}
```

`#pragma omp parallel` *forks* the team of threads. `#pragma omp for` is a
**worksharing** construct that distributes the loop's iterations across the
threads in the surrounding parallel region. The combined form is just a
shortcut when the loop is the entire body of the parallel region.

The split form lets you put code inside the parallel region but outside the
loop. A common use is capturing the team size:

```c
int nthreads = 1;
#pragma omp parallel
{
    nthreads = omp_get_num_threads();    // Inside parallel: returns team size

    #pragma omp for reduction(+:sum)
    for (...) {
        sum += ...;
    }
}
printf("Threads: %d\n", nthreads);       // Back to serial: prints the team size
```

Why split it? `omp_get_num_threads()` only returns the team size when called
**inside a parallel region**. Outside any parallel region it always returns 1,
so there's no place to call it cleanly with the combined `parallel for`.

Every thread writes the same value to the shared `nthreads`, which is safe
because they all write the same value. (`#pragma omp single` is the strictly
canonical "do this once" idiom -- optional for a benign assignment like this.)

### Compiling and Running

```bash
gcc -fopenmp -o program program.c       # Compile with OpenMP
export OMP_NUM_THREADS=4                 # Set thread count
./program                                # Run with 4 threads
```

The environment variable `OMP_NUM_THREADS` controls how many threads are used.
If not set, OpenMP uses all available cores.

---

## Hands-On Exercises (~35 min)

First, navigate to the exercises directory for this module:

```bash
cd module-03-openmp/exercises
```

### Step 0: Look at the Example

Start by examining and running a simple OpenMP hello-world:

```bash
cat ../examples/openmp_hello.c
```

Compile and run it on a compute node:

```bash
gcc -fopenmp -o openmp_hello ../examples/openmp_hello.c

srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 --cpus-per-task=16 \
  bash -c 'export OMP_NUM_THREADS=4; ./openmp_hello'
```

Try changing the thread count (1, 4, 8, 16) and observe the output.

---

### Exercise 1: Understand the Serial Code (Core)

We have a serial program that estimates **pi** using numerical integration.
The idea: the integral of `4 / (1 + x²)` from 0 to 1 equals pi.

```bash
cat pi_serial.c
```

Compile and run it:

```bash
gcc -O2 -o pi_serial pi_serial.c -lm
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./pi_serial
```

Note the computed value of pi and the execution time.

---

### Exercise 2: Add OpenMP Parallelism (Core)

Now open the template with TODO markers:

```bash
cat pi_openmp.c
```

There are **2 TODOs** to fill in:

1. **TODO 1**: Add the OpenMP header include.
2. **TODO 2**: Parallelize the loop using the **split** form of `parallel for`
   so you can also capture the thread count. Specifically: wrap the for loop
   in a `#pragma omp parallel` region, set `nthreads = omp_get_num_threads()`
   inside that region but outside the loop, and put `#pragma omp for
   reduction(+:sum)` on the loop itself. See "Decomposing `parallel for`"
   above for the pattern.

After filling in the TODOs, compile and run:

```bash
gcc -fopenmp -O2 -o pi_openmp pi_openmp.c -lm
```

Then submit the batch script that runs it with varying thread counts:

```bash
sbatch submit_openmp.sh
```

This will run with 1, 2, 4, 8, and 16 threads and report the time for each.
Check the output:

```bash
cat openmp-pi_<JOBID>.out
```

**Questions:**
- Does the answer change with different thread counts? (It shouldn't!)
- How does the time change? What speedup do you get with 16 threads vs. 1?
- Is the speedup perfect (16x)? Why or why not?

---

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

Try asking your AI agent: *"This OpenMP code gives wrong answers. Can you find
the race condition?"* Does it identify the problem correctly?

### Challenge B: Matrix-Vector Multiply

Parallelize a matrix-vector multiplication using OpenMP. A template is at:

```bash
cat matvec_openmp.c
```

The outer loop over rows is embarrassingly parallel -- each row of the output
can be computed independently. Add the appropriate OpenMP directive and compare
performance with the serial version.



---

## Quick Reference

| Directive | Purpose |
|-----------|---------|
| `#pragma omp parallel` | Fork threads (each runs the block) |
| `#pragma omp for` | Distribute loop iterations across threads |
| `#pragma omp parallel for` | Combined: fork + distribute |
| `reduction(+:var)` | Safely sum `var` across threads |
| `private(var)` | Each thread gets its own copy of `var` |
| `omp_get_thread_num()` | Get current thread's ID (0-based) |
| `omp_get_num_threads()` | Get team size (must be called inside a parallel region; returns 1 outside) |

| Environment Variable | Purpose |
|---------------------|---------|
| `OMP_NUM_THREADS=N` | Set number of threads to N |

---

**Next up:** [Module 4 -- Distributed-Memory Parallelism with MPI](../module-04-mpi/README.md)

# Bonus Module -- GPU Offloading with OpenMP

> **Bonus / self-paced module.** This module is not part of the fixed tutorial
> schedule. It builds on [Module 3 (OpenMP)](../module-03-openmp/README.md) and
> offers a directive-based alternative to the explicit GPU programming you saw in
> [Module 5 (HIP)](../module-05-hip/README.md).

## Learning Objectives

By the end of this module, you will be able to:

- Explain the host/device execution model behind GPU offloading
- Offload a compute-intensive loop to the GPU with the `target` construct
- Use `map` clauses to move data between the CPU (host) and GPU (device)
- Split work across the GPU with `teams distribute parallel for` and `collapse`

---

## Key Concepts

### Directive-based GPU programming

OpenMP is a directive-based programming model for shared-memory parallelism --
and since OpenMP 4.5 it can also offload work to GPUs. Instead of a low-level
model like HIP or CUDA, where you explicitly write GPU kernels and manage data
transfers, a directive-based model lets you add *hints* to your existing code
that tell the compiler which regions to run on the GPU and which data to move.

### The host/device model

The CPU and GPU are separate processors with separate memories. The CPU acts
as the **host**: it runs the program until it reaches a compute-intensive region
that can be **offloaded** to the GPU (the **device**). At that point:

```
    Host (CPU)                         Device (GPU)
   ┌──────────┐   1. copy data to      ┌──────────┐
   │  ...run  │ ─────────────────────► │          │
   │  program │   2. compute on GPU    │  kernel  │
   │          │ ◄───────────────────── │          │
   └──────────┘   3. copy results back └──────────┘
```

So to offload work you typically need to (1) transfer data from the host to the
device, (2) run the computation on the device, and (3) transfer results back.

> **Note:** GPUs compute very quickly, so data transfers are often the
> bottleneck in overall run time. Whenever possible, avoid unnecessary
> transfers.

### The `target` construct

`#pragma omp target` marks a region to execute on the GPU:

```c
#pragma omp target
{
    /* this block runs on the device */
}
```

On its own, though, this runs on a *single* GPU thread -- and it does not move
any data.

### Moving data with `map`

`map` clauses tell the compiler which arrays to transfer and in which
direction. The map-type says what to copy:

| map-type | When it copies |
|----------|----------------|
| `to`     | Host → device on entry to the region |
| `from`   | Device → host on exit from the region |
| `tofrom` | Both: copy in on entry, copy back on exit |
| `alloc`  | Allocate on the device, no copy |

Array ranges are written `name[start:count]`. For a matrix multiply
`C = A × B`, `A` and `B` are only read on the GPU while `C` is read and written:

```c
#pragma omp target map(to:A[:N*N], B[:N*N]) map(tofrom:C[:N*N])
```

### Splitting the work across the GPU

To actually use the GPU's parallelism, add the `teams`, `distribute`, and
`parallel for` constructs (usually combined on one line):

- **`teams`** creates a league of thread teams (like a grid of thread blocks)
- **`distribute`** hands out loop iterations across the teams
- **`parallel for`** spreads each team's iterations across its threads

```c
#pragma omp target map(to:A[:N*N], B[:N*N]) map(tofrom:C[:N*N])
#pragma omp teams distribute parallel for collapse(2)
for (int row = 0; row < N; row++) {
    for (int col = 0; col < N; col++) {
        ...
    }
}
```

`collapse(2)` fuses the outer two loops into a single, larger iteration space,
giving the GPU more independent work to distribute.

### Compiling and Running

This cluster's AMD GPUs are compiled for with `amdclang` (from ROCm). The
`--offload-arch` flag targets the MI210's `gfx90a` architecture:

```bash
module load rocm openblas
amdclang -O2 -fopenmp --offload-arch=gfx90a -o program program.c
```

The `openblas` module is only needed for the exercises below, which compare
against a BLAS reference (`-lopenblas`).

---

## Hands-On Exercises

First, navigate to the exercises directory for this module:

```bash
cd openmp-offload/exercises
module load rocm openblas
```

### Step 0: Look at the Example

Start with a tiny program that offloads a loop and reports whether it actually
ran on the GPU:

```bash
cat ../examples/offload_hello.c
```

Compile and run it on a GPU compute node:

```bash
amdclang -O2 -fopenmp --offload-arch=gfx90a -o offload_hello ../examples/offload_hello.c

srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./offload_hello
```

You should see that OpenMP found 1 GPU device and that the target region ran on
the GPU.

---

### Exercise 1: Understand the Serial Code (Core)

`matmul_serial.c` multiplies two N×N matrices two ways: manually in a
triply-nested loop, and with the optimized BLAS call `cblas_dgemm`. The BLAS
result serves as both a correctness check and a speed baseline.

```bash
cat matmul_serial.c
```

Compile and run it:

```bash
amdclang -O2 -fopenmp -o matmul_serial matmul_serial.c \
  -I$OPENBLAS_DIR/include -L$OPENBLAS_DIR/lib -lopenblas -lm

srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./matmul_serial
```

Note the two timings. The manual loop is far slower than the optimized library
call -- that hand-written loop is what we will offload to the GPU next.

---

### Exercise 2: Offload the Loop to the GPU (Core)

Open the template with TODO markers:

```bash
cat matmul_offload.c
```

There are **2 TODOs**, both on the loop nest:

1. **TODO 1**: Add `#pragma omp target` with `map` clauses -- `map(to:...)` for
   the read-only inputs `A` and `B`, and `map(tofrom:...)` for the result `C`.
2. **TODO 2**: Add `teams distribute parallel for collapse(2)` so the work is
   spread across the GPU instead of running on one thread.

See "Splitting the work across the GPU" above for the exact pattern.

After filling in the TODOs, compile:

```bash
amdclang -O2 -fopenmp --offload-arch=gfx90a -o matmul_offload matmul_offload.c \
  -I$OPENBLAS_DIR/include -L$OPENBLAS_DIR/lib -lopenblas -lm
```

Then submit the batch script:

```bash
sbatch submit_offload.sh
```

Check the output once the job finishes:

```bash
cat omp-offload_<JOBID>.out
```

**Questions:**
- Does the offloaded result still match the BLAS reference (0 mismatches)?
- How does the loop time compare to the serial version from Exercise 1?
- How does it compare to the BLAS library time?

A completed version is available at
[`solutions/matmul_offload_solution.c`](solutions/matmul_offload_solution.c).

---

### Challenge A: Experiment with the Directives

Try each of these and re-run, noting the effect on the loop time and
correctness:

- Remove `collapse(2)` -- how much does parallelizing only the outer loop cost?
- Change `map(to:...)` on `C` to `map(tofrom:...)` back and forth, or drop a
  `map` clause entirely -- what happens to correctness?
- Increase `N` (e.g. 2048, 4096) and see how the GPU speedup grows relative to
  the serial loop.

### Challenge B: Ask the Agent

Ask your AI agent: *"I have a serial C matrix-multiply loop. How would I offload
it to a GPU using OpenMP target directives, and what data do I need to map?"*
Compare its suggestion to what you wrote. Does it get the `map` directions right
(`to` for inputs, `tofrom` for the result)? Does it remember `teams distribute
parallel for`?)

---

## Quick Reference

| Directive / Clause | Purpose |
|--------------------|---------|
| `#pragma omp target` | Offload a region to the GPU device |
| `map(to:x[:n])` | Copy `x` host → device on entry |
| `map(from:x[:n])` | Copy `x` device → host on exit |
| `map(tofrom:x[:n])` | Copy `x` in on entry and back on exit |
| `teams` | Create a league of thread teams |
| `distribute` | Spread loop iterations across teams |
| `parallel for` | Spread each team's iterations across its threads |
| `collapse(k)` | Fuse `k` nested loops into one iteration space |

| Function | Purpose |
|----------|---------|
| `omp_get_num_devices()` | Number of GPU devices available |
| `omp_is_initial_device()` | 1 on the host CPU, 0 on the GPU |

| Compile Flag | Purpose |
|--------------|---------|
| `-fopenmp` | Enable OpenMP |
| `--offload-arch=gfx90a` | Target the MI210 GPU architecture |

To learn more, see the [OpenMP specifications](https://www.openmp.org/specifications/).

---

**Back to:** [Tutorial overview](../README.md)

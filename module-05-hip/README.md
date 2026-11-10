# Module 5 -- GPU Programming with HIP

> **Time:** 12:45 to 1:45 PM CST · 60 min total · ~20 min lecture · ~40 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Describe GPU architecture and how it differs from a CPU
- Write a HIP kernel that runs on the AMD MI210 GPU
- Manage data transfers between host (CPU) and device (GPU) memory
- Choose thread and block dimensions for a kernel launch
- Check HIP calls and kernel launches for errors, knowing that launches are asynchronous
- Measure a kernel's performance and explain what limits it
- Check that an agent's experiment script follows its spec and reports failures

---

## Key Concepts

### Why GPUs?

CPUs emphasize low latency and strong performance on individual threads. GPUs
emphasize high throughput by executing many threads in parallel. They work
best when a problem exposes substantial parallelism, such as applying similar
operations to many array elements: vector operations, matrix multiplication,
and neural network layers.

How much faster a GPU is depends on the workload, how it is implemented, the
problem size, and whether you count the time to move data between CPU and GPU
memory. Those copies are slow compared with the GPU's own memory, so real
applications copy their data to the GPU once and keep it there across many
kernels, which reduces the transfer overhead. You'll measure this in
Exercise 1.

### The AMD MI210

Our compute nodes each have an **AMD Instinct MI210** GPU:

| Spec | MI210 |
|------|-------|
| Compute Units | 104 |
| Stream Processors | 6656 |
| Wavefront size | 64 threads |
| Maximum threads per block | 1024 |
| Memory | 64 GB HBM2e |
| Memory Bandwidth | 1.6 TB/s |
| Architecture | CDNA2 (gfx90a) |

### HIP: Portable GPU Programming

**HIP (Heterogeneous-compute Interface for Portability)** is AMD's GPU
programming interface. Its API closely mirrors NVIDIA's CUDA, and HIP code can
be compiled for both AMD and NVIDIA GPUs. Moving code between them can still
take work when it makes assumptions about the hardware (such as the wavefront
size), depends on vendor-specific libraries, or uses a feature that one
platform doesn't support.

### The Programming Model

A HIP program has two parts:

- **Host code** (runs on CPU): manages memory, launches kernels
- **Device code** (runs on GPU): the **kernel** function that runs in parallel

```
  Host (CPU)                    Device (GPU)
  ──────────                    ────────────
  1. Allocate GPU memory        
  2. Copy data to GPU    ──►   
  3. Launch kernel        ──►   Each thread processes one element
  4. Copy results back   ◄──   
  5. Free GPU memory            
```

### Threads, Blocks, and Grids

When you launch a kernel, you specify how many threads to create, organized
into a hierarchy:

```
  Grid (all threads)
  ├── Block 0:  [Thread 0] [Thread 1] ... [Thread 255]
  ├── Block 1:  [Thread 0] [Thread 1] ... [Thread 255]
  ├── Block 2:  [Thread 0] [Thread 1] ... [Thread 255]
  └── ...
```

Each thread computes its **global index** to know which data element to process:

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

| Variable | Meaning |
|----------|---------|
| `threadIdx.x` | Thread index within its block (0 to blockDim-1) |
| `blockIdx.x` | Block index within the grid |
| `blockDim.x` | Number of threads per block |
| `gridDim.x` | Number of blocks in the grid |

**How this maps to the hardware.** On the MI210, threads execute in groups of
64 called **wavefronts**. Each thread block runs entirely on one compute unit,
and a compute unit can host multiple blocks when resources permit. Blocks are
scheduled onto available compute units; a grid can contain many more blocks
than can run simultaneously, and the rest wait for a compute unit to free up.

### Kernel Syntax

```cpp
// Kernel definition (runs on GPU)
__global__ void add_vectors(const double *a, const double *b, double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Launch from host
int threads_per_block = 256;
int blocks = (n + threads_per_block - 1) / threads_per_block;
add_vectors<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
```

The number of blocks is rounded up so that every element gets a thread. When
`n` isn't a multiple of the block size, the last block has more threads than
elements, and the `if (i < n)` check keeps those extra threads from reading or
writing past the end of the arrays.

### Memory Management

| Function | Purpose |
|----------|---------|
| `hipMalloc(&ptr, size)` | Allocate memory on the GPU |
| `hipMemcpy(dst, src, size, kind)` | Copy data between host and device |
| `hipFree(ptr)` | Free GPU memory |
| `hipDeviceSynchronize()` | Wait for all GPU work to finish |

Copy directions:
- `hipMemcpyHostToDevice`: CPU to GPU
- `hipMemcpyDeviceToHost`: GPU to CPU

### Kernel Launches Are Asynchronous

A kernel launch is **asynchronous** with respect to the host: the launch
returns as soon as the work is queued, which does not mean the GPU has
finished running it. The host keeps going until it does something that waits
for the GPU, such as `hipDeviceSynchronize()` or a `hipMemcpy` back to the
host.

This affects how errors show up. Some errors, such as an invalid launch
configuration, are reported by the launch itself. Others happen while the
kernel runs and are reported only when you wait for it. Every HIP call returns
an error code, and ignoring it can hide a failure, so the programs in this
module wrap each call in a small macro, `HIP_CHECK`, that reports which call
failed, where, and why, and exits with a nonzero status:

```cpp
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err_ = (call);                                           \
        if (err_ != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d: %s failed: %s\n",         \
                    __FILE__, __LINE__, #call, hipGetErrorString(err_));    \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

kernel<<<blocks, threads>>>(...);
HIP_CHECK(hipGetLastError());       // Check launch errors
HIP_CHECK(hipDeviceSynchronize()); // Wait and check execution errors
```

Synchronizing right after a launch is useful while learning and debugging,
because it ties each error to the kernel that caused it. It also makes the
host wait, so don't put it inside a loop you are timing: the programs here
check their timed launches with `hipGetLastError()` only.

### What Makes GPU Code Fast

A few rules of thumb explain most GPU performance results, including the ones
you'll measure today:

1. **Give the GPU enough work.** The MI210 has 104 compute units. A kernel
   needs enough blocks and active wavefronts to keep the GPU busy. For
   example, 4,096 threads arranged as 16 blocks of 256 can use at most 16
   compute units at once. Actual utilization also depends on the kernel's
   resource requirements and behavior.
2. **Let neighboring threads read neighboring memory.** When the threads of a
   wavefront read consecutive addresses, the GPU combines their reads into a
   few wide memory transactions (**coalescing**). Scattered reads waste memory
   bandwidth.
3. **Think in wavefronts.** On the MI210, a wavefront is 64 threads (other
   GPUs differ; NVIDIA's equivalent, the warp, has 32). Block sizes that are
   multiples of 64 are a reasonable performance starting point on this GPU,
   not a correctness requirement: a 32-thread block is valid, but it uses only
   half of a wavefront's 64 lanes. The threads of a wavefront execute each
   instruction together, so if some take one side of an `if` and some the
   other, the two paths run one after the other (**divergence**).
4. **Minimize host-device copies.** Copying data between CPU and GPU memory is
   much slower than the GPU's own memory, so keep data on the GPU across as
   many kernels as you can.

### How the Programs Measure Performance

The programs in this module measure performance the same way:

- **Kernel time** comes from GPU events (`hipEventRecord`) recorded just before
  and after the kernel launches, so it covers only the GPU work. Allocation,
  initialization, copies, and validation are outside the timed region. The
  program waits for the stop event (`hipEventSynchronize`) before reading the
  time, so every timed launch has finished.
- **A warm-up launch runs first and isn't timed**, because the first launch in
  a program includes one-time setup.
- **The kernel is then launched 10 times**, and the program reports the
  average time per launch.
- **Effective bandwidth** is the data the algorithm has to move divided by the
  kernel time, in GB/s (10^9 bytes per second). For vector scaling, that is
  2 × n × sizeof(double), for one read of the input and one write of the
  result; vector addition counts two reads and one write. It is calculated
  from the algorithm's nominal data movement, not measured from the GPU's
  memory traffic, so treat the comparison with the MI210's 1.6 TB/s peak as a
  rough guide.

---

## Hands-On Exercises (~40 min)

Suggested pacing: about 7 minutes for Exercise 1, 15 minutes for Exercise 2,
8 minutes for Exercise 3, and 10 minutes for the Agent Practice. Challenges A
and B are optional extensions.

The compiler (`hipcc`) and tools are already set up on the cluster. If
`hipcc --version` doesn't work, run the environment check from
[Getting Started](../README.md#3-verify-your-environment).

First, navigate to the exercises directory for this module:

```bash
cd module-05-hip/exercises
```

### Exercise 1: Run HIP Vector Addition

#### Step 1: Examine the Program

Examine the complete vector addition program:

```bash
cat ../examples/vector_add.cpp
```

#### Step 2: Compile the Program

```bash
hipcc -O2 -o vector_add ../examples/vector_add.cpp
```

#### Step 3: Run on a Compute Node

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./vector_add
```

```{note}
`srun` waits for a free node before it runs your program, so a command may sit
for a while when the partition is busy. Its `--time` limit counts only the time
your program runs, not the time it waits in the queue.
```

#### Step 4: Study the HIP Workflow

Study the code carefully. Note the pattern:
1. Allocate host arrays and initialize them
2. Allocate device arrays with `hipMalloc`
3. Copy input data to device with `hipMemcpy`
4. Launch the kernel with `<<<blocks, threads>>>`, checking for errors
5. Copy results back to host
6. Verify and free memory

#### Step 5: Compare Copy and Kernel Times

The program times the copies as well as the kernel. **Copy to GPU** covers
copying both inputs (`a` and `b`) to the GPU, and **Copy to CPU** covers copying
the result (`c`) back. **Copies + one kernel** adds them to one kernel's time.
It isn't the program's total run time, which also includes allocating and
initializing memory.

**Questions:**
- What fraction of the measured copy-plus-kernel time is spent transferring
  data?
- How might that fraction change if several kernels reused data already on
  the GPU?

---

### Exercise 2: Vector Scale

#### Step 1: Open the Template

Open the exercise template in an editor on the login node:

```bash
vim vector_scale.cpp
```

In Vim, press `i` to enter insert mode and make your edits. Press `Esc`, type
`:wq`, and press `Enter` to save and exit. You can instead use `nano`, `emacs`,
or an existing VS Code Remote SSH connection.

This program should multiply every element of a vector by a constant:
`result[i] = alpha * input[i]`

#### Step 2: Complete the TODOs

There are **4 TODOs** to fill in:

1. **TODO 1**: Write the kernel function
2. **TODO 2**: Copy input data from host to device
3. **TODO 3**: Calculate grid dimensions and launch the kernel
4. **TODO 4**: Copy results from device back to host

TODO 3 checks the launch with `HIP_CHECK(hipGetLastError())`, and the template
then waits with `HIP_CHECK(hipDeviceSynchronize())`: the pattern from
[Kernel Launches Are Asynchronous](#kernel-launches-are-asynchronous).

#### Step 3: Compile the Program

```bash
hipcc -O2 -o vector_scale vector_scale.cpp
```

#### Step 4: Submit the Batch Job

```bash
sbatch submit_hip.sh
```

#### Step 5: Examine the Output

```bash
cat hip-exercise_<JOBID>.out
```

The program checks every element against the expected value (`alpha` times the
input). It prints `PASSED` only if all of them match; otherwise it prints
`FAILED` and the number of mismatches, and exits with a nonzero status. It also
reports the kernel's average time and effective bandwidth (see
[How the Programs Measure Performance](#how-the-programs-measure-performance)),
which you'll compare across block sizes in Exercise 3.

---

### Exercise 3: Experiment with Block Sizes

#### Step 1: Compile One Version per Block Size

`vector_scale.cpp` takes `BLOCK_SIZE` from the compiler command line
(`-DBLOCK_SIZE=...`), so you don't need to edit it. Build one program per
block size from your completed `vector_scale.cpp`. 1024 is the largest block
size the MI210 allows.

```bash
for b in 64 128 256 512 1024; do
  hipcc -O2 -DBLOCK_SIZE=$b -o vector_scale_$b vector_scale.cpp
done
```

The vector length can be set the same way (`-DN=...`). Build one more version
with a length that isn't divisible by the block size, 1,000,003 elements
(3,906 blocks of 256, plus 67), to test the bounds check:

```bash
hipcc -O2 -DN=1000003 -o vector_scale_oddN vector_scale.cpp
```

#### Step 2: Run Them All in One Job

This runs the block sizes twice, to see whether the differences repeat, and
then the odd-length version, all in one job:

```bash
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 \
  bash -c 'for pass in 1 2; do for b in 64 128 256 512 1024; do ./vector_scale_$b; done; done; ./vector_scale_oddN' \
  | grep -E "N:|Block size|bandwidth|Status"
```

**Questions:**
- Does the block size affect correctness? Does the 1,000,003-element version
  pass? In TODO 3 the number of blocks is rounded up, and the kernel checks
  `if (i < n)`: what would go wrong without each of them?
- Are the performance differences consistent across the two passes?
- If smaller blocks are slower, what might explain the result? Possible
  factors include blocks that leave part of a wavefront's lanes idle, limits
  on how many blocks a compute unit can hold at once, and each block's
  register and shared-memory needs. Which of these could apply to a kernel
  this simple?
- Compare the bandwidths with the MI210's 1.6 TB/s peak. Could the kernel
  already have enough parallelism to use the available memory bandwidth
  effectively?

---

### Agent Practice: Automate the Experiment, Honestly

A sweep like Exercise 3's is a natural job for an agent. Ask for a slightly
bigger one that includes a block size you know is too large, and check that
the agent's script follows your spec and reports failures honestly.

#### Step 1: Write the Spec, Then Ask for a Script

Save the specification in a file the agent can read:

```bash
cat > sweep_spec.md << 'EOF'
# sweep.sh specification
- On the login node, compile vector_scale.cpp once for each of these block sizes (-DBLOCK_SIZE): 32, 64, 128, 256, 512, 1024, and 2048.
- Run every configuration that compiled within exactly one srun, on partition mi2101x.
- Keep useful compiler and program error messages.
- For each configuration, report whether it failed to compile, failed to run, or failed validation, and keep going with the rest of the sweep.
- Print the bandwidth only for runs that succeeded and passed validation.
- Never silently skip a failed configuration, and never run a leftover executable from an earlier build.
- Exit with a nonzero status if any configuration failed.
EOF
```

Start Qwen Code:

```bash
bash ../../setup/launch_qwen.sh
```

Ask it:

> Write sweep.sh, which compiles vector_scale.cpp with -DBLOCK_SIZE set to 32, 64, 128, 256, 512, 1024, and 2048, runs all of them in a single srun on our partition, and prints a table of block size, bandwidth, and status.

#### Step 2: Review Before Running

Before you approve the script, answer two questions:
- What should happen at 2048? (See Exercise 3.)
- How many times will this script call `srun`? You asked for one; each extra
  `srun` is another wait in the queue.

If the script doesn't match your spec, tell the agent what to fix before you
approve it.

#### Step 3: Run It and Check

```bash
bash sweep.sh
```

The script compiles on the login node and then runs on a compute node. Check
its table: do the bandwidths match Exercise 3? What does the 2048 row say --
an error, or a blank or plausible-looking number? A script that hides a
failure is worse than one that crashes.

**Questions:**
- What did you have to correct in the agent's script?
- If you had not known that 1024 is the limit, would its table have told you?

**Finish early?** Check the checker: ask the agent, *"Could vector_scale.cpp
print PASSED even if the result were never copied back from the GPU?"* Then
test its answer: comment out the copy back to the host (TODO 4), recompile,
and run it.

---

### Challenge A: Matrix-Vector Multiply on the GPU

Open the challenge template:

```bash
cat matvec_hip.cpp
```

Implement a HIP kernel for matrix-vector multiplication: `y[i] = sum_j(A[i][j] * x[j])`

Each thread computes one element of the output vector y. This means each
thread must iterate over an entire row of the matrix. Name the kernel
`matvec_kernel`, as in the template's TODO. Then compile and run it:

```bash
hipcc -O2 -o matvec_hip matvec_hip.cpp
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./matvec_hip
```

**Questions:**
- Compare the bandwidth with `vector_add`'s from Exercise 1. Why is it so much
  lower? (Hints: how many blocks does the program launch, and how many compute
  units does the MI210 have? At each step of the loop, how far apart in memory
  are the elements that neighboring threads read?)

A completed version is in `solutions/matvec_hip_solution.cpp`.

**Make it faster (optional):** store the matrix transposed, so that element
(i, j) is at `A[j * M + i]`; update the initialization, the CPU reference, and
the kernel. Then at each step of the loop, neighboring threads read
neighboring addresses, and the GPU can combine their reads. How much faster
is it? A completed version is in
`solutions/matvec_hip_transposed_solution.cpp`. The other limit, too few
threads, needs a different design: a whole block per row, combining partial
sums in shared memory, like the reduction in Module 7's capstone.

### Challenge B: Ask the Agent

Give your AI agent this serial C loop and ask it to convert it to a HIP
kernel:

```c
for (int i = 0; i < n; i++) {
    y[i] = 0.0;
    for (int j = 0; j < n; j++) {
        y[i] += A[i * n + j] * x[j];
    }
}
```

Review the generated code. Does the thread indexing make sense? Did it handle
the `if (i < n)` bounds check?

---

## Quick Reference

| HIP Function | Purpose |
|-------------|---------|
| `hipMalloc(&ptr, bytes)` | Allocate GPU memory |
| `hipFree(ptr)` | Free GPU memory |
| `hipMemcpy(dst, src, bytes, dir)` | Copy data (host↔device) |
| `hipDeviceSynchronize()` | Wait for GPU to finish |
| `hipGetLastError()` | Check for kernel launch errors |

| Kernel Syntax | |
|--------------|---|
| `__global__ void kernel(...)` | Declare a GPU kernel |
| `kernel<<<blocks, threads>>>(...)` | Launch the kernel |
| `threadIdx.x` | Thread ID within block |
| `blockIdx.x` | Block ID within grid |
| `blockDim.x` | Threads per block |

| Compile & Run | |
|--------------|---|
| `hipcc -O2 -o prog prog.cpp` | Compile HIP code |
| `srun --partition=mi2101x --nodes=1 --time=2:00 ./prog` | Run on a GPU node |

---

**Next up:** [Module 6 -- AI on HPC: Inference & Fine-Tuning](../module-06-ai-inference-finetuning/README.md)

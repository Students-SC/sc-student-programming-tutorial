# Module 5 -- GPU Programming with HIP

> **Time:** 12:45 to 1:45 PM CST · 60 min total · ~20 min lecture · ~40 min hands-on

## Learning Objectives

By the end of this module, you will be able to:

- Describe GPU architecture and how it differs from a CPU
- Write a HIP kernel that runs on the AMD MI210 GPU
- Manage data transfers between host (CPU) and device (GPU) memory
- Choose thread and block dimensions for a kernel launch

---

## Key Concepts

### Why GPUs?

A CPU has a few powerful cores optimized for complex, sequential tasks. A GPU
has **thousands of simpler cores** optimized for doing the same operation on
many data elements at once.

```
  CPU: 16 powerful cores              GPU: thousands of lightweight cores
  ┌──┐┌──┐┌──┐┌──┐                   ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐
  │  ││  ││  ││  │   ... (16)        │ ││ ││ ││ ││ ││ ││ ││ ││ ││ │ ...
  └──┘└──┘└──┘└──┘                   └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘
                                      (thousands)
```

When your problem involves applying the same math to millions of data points
(vector operations, matrix multiply, neural network layers), GPUs can be
10-100x faster than CPUs.

### The AMD MI210

Our compute nodes each have an **AMD Instinct MI210** GPU:

| Spec | MI210 |
|------|-------|
| Compute Units | 104 |
| Stream Processors | 6656 |
| Memory | 64 GB HBM2e |
| Memory Bandwidth | 1.6 TB/s |
| Architecture | CDNA2 (gfx90a) |

### HIP: Portable GPU Programming

**HIP (Heterogeneous-compute Interface for Portability)** is AMD's GPU
programming framework. It's designed to be nearly identical to NVIDIA's CUDA,
so code is portable across AMD and NVIDIA GPUs.

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

### Memory Management

| Function | Purpose |
|----------|---------|
| `hipMalloc(&ptr, size)` | Allocate memory on the GPU |
| `hipMemcpy(dst, src, size, kind)` | Copy data between host and device |
| `hipFree(ptr)` | Free GPU memory |
| `hipDeviceSynchronize()` | Wait for all GPU work to finish |

Copy directions:
- `hipMemcpyHostToDevice` -- CPU to GPU
- `hipMemcpyDeviceToHost` -- GPU to CPU

---

## Hands-On Exercises (~40 min)

First, navigate to the exercises directory for this module:

```bash
cd module-05-hip/exercises
```

### Step 0: Look at the Example

Examine the complete vector addition program:

```bash
cat ../examples/vector_add.cpp
```

Compile and run it:

```bash
hipcc -O2 -o vector_add ../examples/vector_add.cpp
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./vector_add
```

Study the code carefully. Note the pattern:
1. Allocate host arrays and initialize them
2. Allocate device arrays with `hipMalloc`
3. Copy input data to device with `hipMemcpy`
4. Launch the kernel with `<<<blocks, threads>>>`
5. Copy results back to host
6. Verify and free memory

---

### Exercise 1: Vector Scale (Core)

Open the exercise template:

```bash
cat vector_scale.cpp
```

This program should multiply every element of a vector by a constant:
`result[i] = alpha * input[i]`

There are **4 TODOs** to fill in:

1. **TODO 1**: Write the kernel function
2. **TODO 2**: Copy input data from host to device
3. **TODO 3**: Calculate grid dimensions and launch the kernel
4. **TODO 4**: Copy results from device back to host

After filling in the TODOs, compile and run:

```bash
hipcc -O2 -o vector_scale vector_scale.cpp
sbatch submit_hip.sh
```

Check the output:

```bash
cat hip-exercise_<JOBID>.out
```

The program verifies the results against a CPU reference. If everything is
correct, you'll see "PASSED".

---

### Exercise 2: Experiment with Block Sizes (Core)

Modify `vector_scale.cpp` (or the solution) to try different block sizes.
Edit the `BLOCK_SIZE` define and recompile:

```bash
# Try block sizes: 64, 128, 256, 512
hipcc -O2 -DBLOCK_SIZE=64  -o vector_scale vector_scale.cpp
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./vector_scale

hipcc -O2 -DBLOCK_SIZE=512 -o vector_scale vector_scale.cpp
srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./vector_scale
```

Does the block size affect performance or correctness?

---

### Challenge A: Matrix-Vector Multiply on the GPU

Open the challenge template:

```bash
cat matvec_hip.cpp
```

Implement a HIP kernel for matrix-vector multiplication: `y[i] = sum_j(A[i][j] * x[j])`

Each thread computes one element of the output vector y. This means each
thread must iterate over an entire row of the matrix.

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
| `srun --partition=mi2101x ./prog` | Run on a GPU node |

---

**Next up:** [Module 6 -- AI on HPC: Inference & Fine-Tuning](../module-06-ai-inference-finetuning/README.md)

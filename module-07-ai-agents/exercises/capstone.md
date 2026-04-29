# Capstone Challenge: GPU Dot Product

## The Task

Write a HIP kernel that computes the **dot product** of two vectors using
parallel reduction, run it on the MI210 GPU, and verify the result.

This exercise combines skills from the entire day:

- **Module 2 (Slurm)**: Submit via batch script
- **Module 3 (OpenMP)**: Understanding parallel reduction
- **Module 5 (HIP)**: GPU kernel programming
- **Module 7 (Agents)**: Use your AI agent to help

## What is a Dot Product?

Given two vectors `a` and `b` of length N:

```
dot_product = a[0]*b[0] + a[1]*b[1] + ... + a[N-1]*b[N-1]
```

This is a **reduction** operation -- you need to combine N values into one.

## Why is This Hard on a GPU?

On a CPU, you just loop through the elements. On a GPU with thousands of
threads, you need **parallel reduction**:

```
Step 1:  Thread 0 adds elements 0+1,  Thread 1 adds 2+3,  Thread 2 adds 4+5, ...
Step 2:  Thread 0 adds results 0+1,  Thread 1 adds 2+3, ...
Step 3:  Thread 0 adds results 0+1, ...
...
Final:   Thread 0 has the total
```

This is a tree-shaped reduction that takes O(log N) steps instead of O(N).

## Steps

1. **Generate the code.** Ask your agent or AI assistant:
   *"Write a HIP program that computes the dot product of two float vectors
   using parallel reduction in shared memory. Use 256 threads per block.
   Verify against a CPU reference."*

2. **Review the code.** Before compiling, check:
   - Does the kernel use `__shared__` memory for the reduction?
   - Is there a `__syncthreads()` between reduction steps?
   - Does it handle the case where N is not a multiple of the block size?
   - Does it handle multiple blocks (each block produces a partial sum)?

3. **Compile and run.**
   ```bash
   hipcc -O2 -o dot_product dot_product.cpp
   srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./dot_product
   ```

4. **Verify.** Does the GPU result match the CPU reference within floating-point
   tolerance?

## Bonus Challenges

- Compare performance of your GPU dot product to a CPU version
- Try different vector sizes (1M, 10M, 100M elements)
- Implement the reduction using `atomicAdd` instead of shared memory -- is it
  faster or slower?
- Use `hipEventRecord` to time just the kernel (excluding memory transfers)

## Hints

- Shared memory declaration: `__shared__ float sdata[BLOCK_SIZE];`
- Synchronize threads in a block: `__syncthreads();`
- The standard reduction pattern halves the active threads each step:
  ```cpp
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (threadIdx.x < s)
          sdata[threadIdx.x] += sdata[threadIdx.x + s];
      __syncthreads();
  }
  ```
- Each block writes its partial sum to a global array, then you sum those
  on the CPU (or launch a second kernel).

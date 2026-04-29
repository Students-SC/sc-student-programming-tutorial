/*
 * vector_scale.cpp -- Scale a vector on the GPU: result[i] = alpha * input[i]
 *
 * EXERCISE: Fill in the 4 TODOs to complete this HIP program.
 *
 * Compile:  hipcc -O2 -o vector_scale vector_scale.cpp
 * Run:      srun --partition=mi2101x --time=2:00 --ntasks=1 ./vector_scale
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define N (1 << 24)   /* ~16 million elements */

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, hipGetErrorString(err));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


/* TODO 1: Write the kernel function.
 *
 * It should:
 *   - Be marked __global__
 *   - Take parameters: const double *input, double *result, double alpha, int n
 *   - Compute the global thread index: i = blockIdx.x * blockDim.x + threadIdx.x
 *   - If i < n, set result[i] = alpha * input[i]
 *
 * YOUR CODE HERE:
 */


int main(void) {
    double alpha = 2.5;
    size_t bytes = N * sizeof(double);

    /* Allocate and initialize host memory */
    double *h_input  = (double *)malloc(bytes);
    double *h_result = (double *)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_input[i] = (double)i;
    }

    /* Allocate device memory */
    double *d_input, *d_result;
    HIP_CHECK(hipMalloc(&d_input, bytes));
    HIP_CHECK(hipMalloc(&d_result, bytes));

    /* TODO 2: Copy h_input to d_input using hipMemcpy.
     *
     *   HIP_CHECK(hipMemcpy(d_input, h_input, bytes, ???));
     *
     * Which direction? hipMemcpyHostToDevice or hipMemcpyDeviceToHost?
     *
     * YOUR CODE HERE:
     */

    /* TODO 3: Calculate the number of blocks and launch the kernel.
     *
     *   int blocks = ???;
     *   vector_scale_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_result, alpha, N);
     *   HIP_CHECK(hipGetLastError());
     *
     * Hint: blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE   (round up)
     *
     * YOUR CODE HERE:
     */

    HIP_CHECK(hipDeviceSynchronize());

    /* TODO 4: Copy d_result back to h_result using hipMemcpy.
     *
     *   HIP_CHECK(hipMemcpy(h_result, d_result, bytes, ???));
     *
     * YOUR CODE HERE:
     */

    /* Verify results */
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double expected = alpha * (double)i;
        double error = fabs(h_result[i] - expected);
        if (error > max_error) max_error = error;
    }

    printf("Vector Scale: result[i] = %.1f * input[i]\n", alpha);
    printf("  N:          %d\n", N);
    printf("  Block size: %d\n", BLOCK_SIZE);
    printf("  Max error:  %.2e\n", max_error);
    printf("  Status:     %s\n", max_error < 1e-10 ? "PASSED" : "FAILED");

    /* Clean up */
    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_result));
    free(h_input);
    free(h_result);

    return 0;
}

/*
 * vector_add.cpp -- Vector addition on the GPU using HIP
 *
 * Computes c[i] = a[i] + b[i] for N elements.
 *
 * Compile:  hipcc -O2 -o vector_add vector_add.cpp
 * Run:      srun --partition=mi2101x --time=2:00 --ntasks=1 ./vector_add
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cmath>

#define N (1 << 24)   /* ~16 million elements */
#define BLOCK_SIZE 256

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, hipGetErrorString(err));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* ---- Kernel: each thread adds one pair of elements ---- */
__global__ void vector_add_kernel(const double *a, const double *b,
                                  double *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(void) {
    size_t bytes = N * sizeof(double);

    /* Allocate host memory */
    double *h_a = (double *)malloc(bytes);
    double *h_b = (double *)malloc(bytes);
    double *h_c = (double *)malloc(bytes);

    /* Initialize input arrays */
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
    }

    /* Allocate device (GPU) memory */
    double *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    /* Copy inputs from host to device */
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));

    /* Launch kernel */
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching kernel: %d blocks x %d threads = %d total threads\n",
           blocks, BLOCK_SIZE, blocks * BLOCK_SIZE);
    printf("Problem size: %d elements (%.1f MB)\n\n", N, bytes / 1e6);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    vector_add_kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float kernel_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));

    /* Copy results back to host */
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));

    /* Verify: sin^2(x) + cos^2(x) should equal 1.0 */
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(h_c[i] - 1.0);
        if (error > max_error) max_error = error;
    }

    printf("Results:\n");
    printf("  Kernel time:  %.3f ms\n", kernel_ms);
    printf("  Bandwidth:    %.1f GB/s (kernel only)\n",
           3.0 * bytes / (kernel_ms / 1e3) / 1e9);
    printf("  Max error:    %.2e\n", max_error);
    printf("  Status:       %s\n", max_error < 1e-10 ? "PASSED" : "FAILED");

    /* Clean up */
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}

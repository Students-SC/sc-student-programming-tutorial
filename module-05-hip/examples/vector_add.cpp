/*
 * vector_add.cpp -- Vector addition on the GPU using HIP
 *
 * Computes c[i] = a[i] + b[i] for N elements.
 *
 * Compile:  hipcc -O2 -o vector_add vector_add.cpp
 * Run:      srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./vector_add
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define N (1 << 24)   /* ~16 million elements */
#define BLOCK_SIZE 256
#define REPS 10       /* timed kernel launches, averaged */

/* Check a HIP call's return value. On failure, report which call failed,
 * where, and why, then exit with a nonzero status. */
#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err_ = (call);                                           \
        if (err_ != hipSuccess) {                                           \
            fprintf(stderr, "HIP error at %s:%d: %s failed: %s\n",         \
                    __FILE__, __LINE__, #call, hipGetErrorString(err_));    \
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

    /* Initialize the inputs. Fill the output with NaN so that validation
     * fails if no result is ever copied back. */
    for (int i = 0; i < N; i++) {
        h_a[i] = sin(i) * sin(i);
        h_b[i] = cos(i) * cos(i);
        h_c[i] = NAN;
    }

    /* Allocate device (GPU) memory */
    double *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    /* Copy the inputs from host to device (timed) */
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(d_a, h_a, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b, bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float h2d_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&h2d_ms, start, stop));

    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Launching kernel: %d blocks x %d threads = %d total threads\n",
           blocks, BLOCK_SIZE, blocks * BLOCK_SIZE);
    printf("Problem size: %d elements (%.1f MB per array)\n\n", N, bytes / 1e6);

    /* Warm-up launch. The first launch in a program includes one-time setup,
     * so it is not timed. Launches return before the GPU finishes, so check
     * for launch errors first, then wait and check for errors while it ran. */
    vector_add_kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    /* Time REPS launches with GPU events and report the average. */
    HIP_CHECK(hipEventRecord(start));
    for (int rep = 0; rep < REPS; rep++)
        vector_add_kernel<<<blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    float kernel_ms = total_ms / REPS;

    /* Copy the result back to the host (timed) */
    HIP_CHECK(hipEventRecord(start));
    HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float d2h_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&d2h_ms, start, stop));

    /* Validate: sin^2(x) + cos^2(x) should equal 1.0. NaN fails every
     * comparison, so reject non-finite values explicitly. */
    int mismatches = 0;
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double error = fabs(h_c[i] - 1.0);
        if (!std::isfinite(h_c[i]) || error > 1e-10) mismatches++;
        else if (error > max_error) max_error = error;
    }
    int passed = (mismatches == 0);

    printf("Results:\n");
    printf("  Kernel time:          %.3f ms (average of %d launches)\n", kernel_ms, REPS);
    printf("  Effective bandwidth:  %.1f GB/s (reads a and b, writes c)\n",
           3.0 * bytes / (kernel_ms / 1e3) / 1e9);
    printf("  Copy to GPU:          %.3f ms (a and b)\n", h2d_ms);
    printf("  Copy to CPU:          %.3f ms (c)\n", d2h_ms);
    printf("  Copies + one kernel:  %.3f ms\n", h2d_ms + kernel_ms + d2h_ms);
    printf("  Max error:            %.2e\n", max_error);
    printf("  Mismatches:           %d of %d\n", mismatches, N);
    printf("  Status:               %s\n", passed ? "PASSED" : "FAILED");

    /* Clean up */
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);

    return passed ? 0 : 1;
}

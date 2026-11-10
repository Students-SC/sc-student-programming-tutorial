/*
 * histogram_checked_solution.cpp -- Reference solution: a 16-bin histogram
 * on the GPU, checked bin by bin (Agent Practice)
 *
 * Compile:  hipcc -O2 -o histogram_checked histogram_checked_solution.cpp
 * Run:      srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=1 ./histogram_checked
 *
 * Optional compile-time setting:
 *   -DN=<number of elements>           (default 1000003)
 *
 * Instructor note: students ask Qwen Code to write their own
 * histogram_basic.cpp and histogram_checked.cpp, so no student versions are
 * provided. If a student's histogram_checked.cpp still doesn't work after one
 * or two fixes, have them copy this file in its place and continue with the
 * comparison:
 *     cp solutions/histogram_checked_solution.cpp histogram_checked.cpp
 *
 * In Step 6's deliberate defect, the kernel adds to bin (input[i] + 1) % 16
 * instead of bin input[i]. Every count moves up one bin, so the total stays
 * 1,000,003, but bin 0 gets 62,500 instead of 62,501 and bin 3 gets 62,501
 * instead of 62,500. The program reports 2 mismatched bins, prints FAILED,
 * and exits with status 1, so srun also reports the task's nonzero exit
 * code. Both are expected.
 *
 * For the closing question about other inputs to test, possible answers
 * include all elements equal to 0, all equal to 15, fewer elements than bins
 * (try -DN=10: bins 0 to 9 get 1 each), and a length that isn't a multiple of
 * the block size (1,000,003 already isn't).
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef N
#define N 1000003
#endif

#define NUM_BINS 16
#define BLOCK_SIZE 256

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

/* One thread per input element. Many threads add to the same bin at the
 * same time, so each addition must be atomic or some would be lost. */
__global__ void histogram_kernel(const int *input, int *histogram, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&histogram[input[i]], 1);
    }
}

int main(void) {
    size_t input_bytes = (size_t)N * sizeof(int);
    size_t hist_bytes = NUM_BINS * sizeof(int);

    /* Initialize the input. Fill the host histogram with -1 so that
     * validation fails if no result is ever copied back. */
    int *h_input = (int *)malloc(input_bytes);
    int h_histogram[NUM_BINS];
    for (int i = 0; i < N; i++) {
        h_input[i] = i % NUM_BINS;
    }
    for (int b = 0; b < NUM_BINS; b++) {
        h_histogram[b] = -1;
    }

    int *d_input, *d_histogram;
    HIP_CHECK(hipMalloc(&d_input, input_bytes));
    HIP_CHECK(hipMalloc(&d_histogram, hist_bytes));
    HIP_CHECK(hipMemcpy(d_input, h_input, input_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_histogram, 0, hist_bytes));

    /* Round the number of blocks up so the last, partial block is launched;
     * the kernel's bounds check skips its extra threads. */
    int blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram_kernel<<<blocks, BLOCK_SIZE>>>(d_input, d_histogram, N);
    HIP_CHECK(hipGetLastError());

    /* The launch returns before the GPU finishes. Wait for it, and check for
     * errors that happened while the kernel ran. */
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_histogram, d_histogram, hist_bytes, hipMemcpyDeviceToHost));

    /* The expected counts come from N and NUM_BINS alone: every bin gets
     * N / NUM_BINS elements, and the first N % NUM_BINS bins get one more. */
    long long expected_total = 0, actual_total = 0;
    int mismatches = 0;
    printf("Histogram of N = %d integers, input[i] = i %% %d\n\n", N, NUM_BINS);
    printf("Bin   Expected   Actual   Match\n");
    for (int b = 0; b < NUM_BINS; b++) {
        int expected = N / NUM_BINS + (b < N % NUM_BINS ? 1 : 0);
        int match = (h_histogram[b] == expected);
        if (!match) mismatches++;
        expected_total += expected;
        actual_total += h_histogram[b];
        printf("%3d   %8d   %6d   %s\n", b, expected, h_histogram[b], match ? "yes" : "NO");
    }
    int passed = (mismatches == 0 && actual_total == expected_total);

    printf("\n");
    printf("Expected total:  %lld\n", expected_total);
    printf("Actual total:    %lld\n", actual_total);
    printf("Mismatched bins: %d\n", mismatches);
    printf("Overall:         %s\n", passed ? "PASSED" : "FAILED");

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_histogram));
    free(h_input);

    return passed ? 0 : 1;
}

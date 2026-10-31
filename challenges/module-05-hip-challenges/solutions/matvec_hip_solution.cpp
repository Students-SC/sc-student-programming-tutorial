/*
 * matvec_hip_solution.cpp -- Solution: matrix-vector multiply on the GPU
 *
 * Compile:  hipcc -O2 -o matvec_hip matvec_hip_solution.cpp
 * Run:      srun --partition=mi2101x --time=2:00 --ntasks=1 ./matvec_hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define M 4096
#define K 4096
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

__global__ void matvec_kernel(const double *A, const double *x, double *y,
                              int m, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += A[row * k + j] * x[j];
        }
        y[row] = sum;
    }
}

int main(void) {
    size_t bytes_A = (size_t)M * K * sizeof(double);
    size_t bytes_x = K * sizeof(double);
    size_t bytes_y = M * sizeof(double);

    double *h_A = (double *)malloc(bytes_A);
    double *h_x = (double *)malloc(bytes_x);
    double *h_y = (double *)malloc(bytes_y);
    double *h_ref = (double *)malloc(bytes_y);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = (double)(i + j) / K;
    for (int j = 0; j < K; j++)
        h_x[j] = 1.0;

    for (int i = 0; i < M; i++) {
        h_ref[i] = 0.0;
        for (int j = 0; j < K; j++)
            h_ref[i] += h_A[i * K + j] * h_x[j];
    }

    double *d_A, *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_A, bytes_A));
    HIP_CHECK(hipMalloc(&d_x, bytes_x));
    HIP_CHECK(hipMalloc(&d_y, bytes_y));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, bytes_x, hipMemcpyHostToDevice));

    int blocks = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    matvec_kernel<<<blocks, BLOCK_SIZE>>>(d_A, d_x, d_y, M, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float kernel_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));

    HIP_CHECK(hipMemcpy(h_y, d_y, bytes_y, hipMemcpyDeviceToHost));

    double max_error = 0.0;
    for (int i = 0; i < M; i++) {
        double error = fabs(h_y[i] - h_ref[i]);
        if (error > max_error) max_error = error;
    }

    printf("Matrix-Vector Multiply (%d x %d) on GPU\n", M, K);
    printf("  Kernel time:  %.3f ms\n", kernel_ms);
    printf("  Max error:    %.2e\n", max_error);
    printf("  Status:       %s\n", max_error < 1e-6 ? "PASSED" : "FAILED");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    free(h_A);
    free(h_x);
    free(h_y);
    free(h_ref);

    return 0;
}

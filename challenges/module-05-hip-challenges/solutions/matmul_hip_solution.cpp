/*
 * matmul_hip_solution.cpp -- Solution: matrix-matrix multiply on the GPU
 *
 * Compile:  hipcc -O2 -o matmul_hip matmul_hip_solution.cpp
 * Run:      srun --partition=mi2101x --time=2:00 --ntasks=1 ./matmul_hip
 */

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define M 1024
#define K 1024
#define N 1024
#define BLOCK_SIZE 16

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = call;                                              \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, hipGetErrorString(err));             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__global__ void matmul_kernel(const double *A, const double *B, double *C,
                              int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        double sum = 0.0;
        for (int j = 0; j < k; j++) {
            sum += A[row * k + j] * B[j * n + col];
        }
        C[row * n + col] = sum;
    }
}

int main(void) {
    size_t bytes_A = (size_t)M * K * sizeof(double);
    size_t bytes_B = (size_t)K * N * sizeof(double);
    size_t bytes_C = (size_t)M * N * sizeof(double);

    double *h_A = (double *)malloc(bytes_A);
    double *h_B = (double *)malloc(bytes_B);
    double *h_C = (double *)malloc(bytes_C);
    double *h_ref = (double *)malloc(bytes_C);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < K; j++)
            h_A[i * K + j] = (double)(i + j) / K;
    for (int j = 0; j < K; j++)
        for (int c = 0; c < N; c++)
            h_B[j * N + c] = 1.0;

    for (int i = 0; i < M; i++) {
        for (int c = 0; c < N; c++) {
            h_ref[i * N + c] = 0.0;
            for (int j = 0; j < K; j++)
                h_ref[i * N + c] += h_A[i * K + j] * h_B[j * N + c];
        }
    }

    double *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes_A));
    HIP_CHECK(hipMalloc(&d_B, bytes_B));
    HIP_CHECK(hipMalloc(&d_C, bytes_C));

    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes_B, hipMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float kernel_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));

    HIP_CHECK(hipMemcpy(h_C, d_C, bytes_C, hipMemcpyDeviceToHost));

    double max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        double error = fabs(h_C[i] - h_ref[i]);
        if (error > max_error) max_error = error;
    }

    printf("Matrix-Matrix Multiply (%d x %d) * (%d x %d) on GPU\n", M, K, K, N);
    printf("  Kernel time:  %.3f ms\n", kernel_ms);
    printf("  Max error:    %.2e\n", max_error);
    printf("  Status:       %s\n", max_error < 1e-6 ? "PASSED" : "FAILED");

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_ref);

    return 0;
}

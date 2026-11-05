/*
 * matmul_hip_solution.cpp -- Solution: matrix-matrix multiply on the GPU
 *
 * We will use 2D instead of 1D thread blocks.
 *
 * Compile:  hipcc -O2 -o matmul_hip matmul_hip_solution.cpp
 * Run:      srun --partition=mi2101x --time=2:00 --ntasks=1 ./matmul_hip
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>

#define M 1024
#define K 1024
#define N 1024
#define BLOCK_DIM 16

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,          \
              hipGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

/* TODO: write the matmul_kernel
 * Each thread computes one element of C given the row index of A and col index
 * of B: C[row * n + col] = sum over j of A[row * K + j] * B[j * N + col]
 *
 * Parameters: const double *A, const double *B, double *C, int m, int k, int n
 *
 * Steps:
 * 1. Compute the row index and col index from thread/block IDs. Hint: you can
 * blockIdx.y, blockDim.y, threadIdx.y since the blocks are 2D.
 * 2. If row < m and col < n, loop over columns j = 0..k-1
 * accumulating A[row * K + j] * B[j * N + col]
 * 3. Store the result in C[row * n + col]
 *
 * __global__ void matmul_kernel() {}
 *
 *
 *
 */

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

  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (M + BLOCK_DIM - 1) / BLOCK_DIM);

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));
  /* TODO: Launch the kernel
   * matmul_kernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
   */
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float kernel_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&kernel_ms, start, stop));

  HIP_CHECK(hipMemcpy(h_C, d_C, bytes_C, hipMemcpyDeviceToHost));

  double max_error = 0.0;
  for (int i = 0; i < M * N; i++) {
    double error = fabs(h_C[i] - h_ref[i]);
    if (error > max_error)
      max_error = error;
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

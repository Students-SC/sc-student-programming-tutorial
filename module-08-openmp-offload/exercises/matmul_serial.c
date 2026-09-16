/*
 * matmul_serial.c -- Matrix multiply, serial reference version
 *
 * Multiplies two N x N matrices of random numbers two ways:
 *   1. Manually, with a triply-nested loop (this is what we will offload later)
 *   2. With the optimized BLAS call cblas_dgemm (correctness + speed baseline)
 *
 * The BLAS result is used both to check our loop's answer and to give a
 * time-to-solution to compare against.
 *
 * Compile (needs OpenBLAS -- see the README):
 *   amdclang -O2 -fopenmp -o matmul_serial matmul_serial.c \
 *     -I$OPENBLAS_DIR/include -L$OPENBLAS_DIR/lib -lopenblas -lm
 * Run:
 *   ./matmul_serial
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>
#include <omp.h>

/* Matrix dimension. Kept modest so the serial loop finishes in seconds. */
#define N 1024

int main(void) {
    double start_total = omp_get_wtime();
    double start, stop, loop_elapsed_time, library_elapsed_time;

    size_t buffer_size = (size_t)N * N * sizeof(double);
    double *A     = (double *)malloc(buffer_size);
    double *B     = (double *)malloc(buffer_size);
    double *C     = (double *)malloc(buffer_size);
    double *ref_C = (double *)malloc(buffer_size);

    /* Fill A and B with random values; zero out C and ref_C */
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int index = row * N + col;
            A[index]     = (double)rand() / (double)RAND_MAX;
            B[index]     = (double)rand() / (double)RAND_MAX;
            C[index]     = 0.0;
            ref_C[index] = 0.0;
        }
    }

    /* --- Manual matrix multiply in a loop --- */
    start = omp_get_wtime();
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            for (int k = 0; k < N; k++) {
                int index = row * N + col;
                C[index] = C[index] + A[row * N + k] * B[k * N + col];
            }
        }
    }
    stop = omp_get_wtime();
    loop_elapsed_time = stop - start;

    /* --- Same multiply via the BLAS library (reference) --- */
    const double alpha = 1.0;
    const double beta  = 0.0;
    double tolerance   = 1.0e-10;

    start = omp_get_wtime();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, alpha, A, N, B, N, beta, ref_C, N);
    stop = omp_get_wtime();
    library_elapsed_time = stop - start;

    /* --- Compare our loop's result against the BLAS reference --- */
    int errors = 0;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            int index = row * N + col;
            if (fabs(C[index] - ref_C[index]) > tolerance) errors++;
        }
    }

    double total_elapsed_time = omp_get_wtime() - start_total;

    printf("Matrix size             : %d x %d\n", N, N);
    printf("Mismatches vs. reference : %d (should be 0)\n", errors);
    printf("Elapsed time total (s)  : %16.14f\n", total_elapsed_time);
    printf("Elapsed time loop (s)   : %16.14f\n", loop_elapsed_time);
    printf("Elapsed time library (s): %16.14f\n", library_elapsed_time);

    free(A); free(B); free(C); free(ref_C);
    return 0;
}

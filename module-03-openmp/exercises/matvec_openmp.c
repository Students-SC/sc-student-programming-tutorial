/*
 * matvec_openmp.c -- Matrix-vector multiplication with OpenMP
 *
 * CHALLENGE: Add OpenMP directives to parallelize the matrix-vector multiply.
 *
 * The program multiplies a matrix A (N x N) by a vector x, producing y = A*x.
 * It first runs a serial version for reference, then you parallelize the
 * second version and compare times.
 *
 * Compile:  gcc -fopenmp -O2 -o matvec_openmp matvec_openmp.c -lm
 * Run:      OMP_NUM_THREADS=8 ./matvec_openmp
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define N 8000

static double A[N][N];
static double x[N];
static double y_serial[N];
static double y_parallel[N];

int main(void) {
    /* Initialize matrix and vector with simple values */
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / N;
        }
    }

    /* --- Serial version (reference) --- */
    double t_start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        y_serial[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_serial[i] += A[i][j] * x[j];
        }
    }
    double t_serial = omp_get_wtime() - t_start;

    /* --- Parallel version (YOUR WORK) --- */
    /*
     * TODO: Add an OpenMP directive to parallelize the OUTER loop (over i).
     * Each row i is independent -- its output y_parallel[i] depends only
     * on row i of A and the vector x.
     *
     * Hint: Which loop should you parallelize? The outer one (i) or
     * the inner one (j)?  Think about what each thread would write to.
     */
    t_start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        y_parallel[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_parallel[i] += A[i][j] * x[j];
        }
    }
    double t_parallel = omp_get_wtime() - t_start;

    /* --- Verify correctness --- */
    double max_diff = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs(y_serial[i] - y_parallel[i]);
        if (diff > max_diff) max_diff = diff;
    }

    printf("Matrix-Vector Multiply (%d x %d)\n", N, N);
    printf("Serial time:    %.3f seconds\n", t_serial);
    printf("Parallel time:  %.3f seconds\n", t_parallel);
    printf("Speedup:        %.2fx\n", t_serial / t_parallel);
    printf("Threads:        %d\n", omp_get_max_threads());
    printf("Max difference: %.2e (should be ~0)\n", max_diff);

    return 0;
}

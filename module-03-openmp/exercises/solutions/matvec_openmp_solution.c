/*
 * matvec_openmp_solution.c -- Solution: matrix-vector multiply with OpenMP
 *
 * Compile:  gcc -fopenmp -O2 -o matvec_openmp matvec_openmp_solution.c -lm
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
    for (int i = 0; i < N; i++) {
        x[i] = 1.0;
        for (int j = 0; j < N; j++) {
            A[i][j] = (double)(i + j) / N;
        }
    }

    /* Serial version */
    double t_start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        y_serial[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_serial[i] += A[i][j] * x[j];
        }
    }
    double t_serial = omp_get_wtime() - t_start;

    /* Parallel version -- parallelize the outer loop */
    t_start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        y_parallel[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_parallel[i] += A[i][j] * x[j];
        }
    }
    double t_parallel = omp_get_wtime() - t_start;

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

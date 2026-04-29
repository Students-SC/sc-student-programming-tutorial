/*
 * pi_openmp_solution.c -- Solution: pi estimation with OpenMP
 *
 * Compile:  gcc -fopenmp -O2 -o pi_openmp pi_openmp_solution.c -lm
 * Run:      OMP_NUM_THREADS=4 ./pi_openmp
 */

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>            /* TODO 1: SOLVED */

#define NUM_STEPS 100000000L

int main(void) {
    double step = 1.0 / (double)NUM_STEPS;
    double sum = 0.0;
    int nthreads = 1;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* TODO 2: SOLVED -- split form of "parallel for" so we can capture
     * nthreads inside the parallel region but outside the loop. */
    #pragma omp parallel
    {
        nthreads = omp_get_num_threads();

        #pragma omp for reduction(+:sum)
        for (long i = 0; i < NUM_STEPS; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }
    }

    double pi = sum * step;

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = (t_end.tv_sec - t_start.tv_sec)
                   + (t_end.tv_nsec - t_start.tv_nsec) / 1e9;

    printf("Computed pi = %.15f\n", pi);
    printf("Reference   = %.15f\n", M_PI);
    printf("Error       = %.2e\n", fabs(pi - M_PI));
    printf("Steps       = %ld\n", NUM_STEPS);
    printf("Threads     = %d\n", nthreads);
    printf("Time        = %.3f seconds\n", elapsed);

    return 0;
}

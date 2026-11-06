/*
 * pi_race_solution.c -- Solution: the race condition fixed with a reduction
 *
 * Compile:  gcc -fopenmp -O2 -o pi_race pi_race_solution.c -lm
 * Run:      OMP_NUM_THREADS=8 ./pi_race
 */

#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NUM_STEPS 1000000000L

int main(void) {
    double step = 1.0 / (double)NUM_STEPS;
    double sum = 0.0;

    double t_start = omp_get_wtime();

    /* FIXED: reduction(+:sum) gives each thread a private copy of 'sum'
     * and adds the copies together at the end, so no two threads write
     * the shared 'sum' at the same time. */
    #pragma omp parallel for reduction(+:sum)
    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = sum * step;
    double elapsed = omp_get_wtime() - t_start;

    printf("Computed pi = %.15f\n", pi);
    printf("Reference   = %.15f\n", M_PI);
    printf("Error       = %.2e\n", fabs(pi - M_PI));
    printf("Threads     = %d\n", omp_get_max_threads());
    printf("Time        = %.3f seconds\n", elapsed);

    if (fabs(pi - M_PI) > 0.001) {
        printf("\n*** WARNING: Result is way off! There's a race condition. ***\n");
    }

    return 0;
}

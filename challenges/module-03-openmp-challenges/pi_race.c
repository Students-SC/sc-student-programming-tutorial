/*
 * pi_race.c -- Pi calculation with a DELIBERATE race condition
 *
 * CHALLENGE: This code has a bug! It gives different (wrong) answers
 * each time you run it with multiple threads. Can you find and fix it?
 *
 * Compile:  gcc -fopenmp -O2 -o pi_race pi_race.c -lm
 * Run:      OMP_NUM_THREADS=8 ./pi_race
 */

#include <stdio.h>
#include <math.h>
#include <omp.h>

#define NUM_STEPS 100000000L

int main(void) {
    double step = 1.0 / (double)NUM_STEPS;
    double sum = 0.0;

    double t_start = omp_get_wtime();

    /*
     * BUG: The parallel for is missing a reduction clause.
     * Multiple threads are writing to 'sum' simultaneously,
     * causing a race condition.
     */
    #pragma omp parallel for
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

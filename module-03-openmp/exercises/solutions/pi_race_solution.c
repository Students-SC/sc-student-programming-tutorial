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
     * the shared 'sum' at the same time.
     *
     * Why were the wrong answers so small? Without the reduction, each
     * thread read and updated the one shared 'sum'. With -O2 the compiler
     * keeps a thread's running total in a register and writes it back to
     * 'sum' when that thread finishes its share of the loop, so the last
     * thread to finish overwrites everyone else's work. The "answer" is
     * then roughly one thread's share of pi; with 2 threads, for example,
     * it is about 1.85, the integral over the first half of [0, 1]. If the
     * same thread finishes last on every run, the same wrong answer
     * repeats, which is why it doesn't always change between runs. */
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

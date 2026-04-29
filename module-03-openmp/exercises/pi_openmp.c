/*
 * pi_openmp.c -- Estimate pi using numerical integration (OpenMP)
 *
 * EXERCISE: Fill in the 2 TODOs to parallelize this program with OpenMP.
 *
 * Compile:  gcc -fopenmp -O2 -o pi_openmp pi_openmp.c -lm
 * Run:      OMP_NUM_THREADS=4 ./pi_openmp
 */

#include <stdio.h>
#include <math.h>
#include <time.h>

/* TODO 1: Include the OpenMP header file.
 *
 *   #include <???>
 */

#define NUM_STEPS 100000000L

int main(void) {
    double step = 1.0 / (double)NUM_STEPS;
    double sum = 0.0;
    int nthreads = 1;

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    /* TODO 2: Parallelize the for loop AND capture the thread count.
     *
     * Use the SPLIT form of "parallel for" -- see the README's
     * "Decomposing parallel for" section for the pattern. Specifically:
     *
     *   1. Wrap the for loop in a parallel region:
     *        #pragma omp parallel
     *        {
     *            ...
     *        }
     *
     *   2. Inside that region but OUTSIDE the for loop, set:
     *        nthreads = omp_get_num_threads();
     *
     *   3. Put a worksharing-for directive on the loop itself, with a
     *      reduction clause to safely accumulate 'sum':
     *        #pragma omp for reduction(+:sum)
     *
     * Why split it?  omp_get_num_threads() only returns the team size
     * when called inside a parallel region -- outside, it always returns
     * 1.  The combined "#pragma omp parallel for" gives you no place to
     * call it.
     */
    for (long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
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

/*
 * openmp_hello.c -- OpenMP Hello World
 *
 * Each thread prints its ID and the total number of threads.
 *
 * Compile:  gcc -fopenmp -o openmp_hello openmp_hello.c
 * Run:      OMP_NUM_THREADS=4 ./openmp_hello
 */

#include <stdio.h>
#include <omp.h>

int main(void) {
    printf("Before the parallel region: 1 thread\n\n");

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("  Hello from thread %2d of %d\n", tid, nthreads);
    }

    printf("\nBack to 1 thread after the parallel region.\n");
    return 0;
}

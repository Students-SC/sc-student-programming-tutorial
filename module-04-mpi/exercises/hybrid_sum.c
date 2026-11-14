/*
 * hybrid_sum.c -- Parallel sum with MPI ranks AND OpenMP threads
 *
 * CHALLENGE: Fill in the 2 TODOs so that each MPI rank splits its share of
 * the work across OpenMP threads.
 *
 * Compile:  mpicc -fopenmp -O2 -o hybrid_sum hybrid_sum.c -lm
 * Run:      sbatch submit_hybrid.sh
 */

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define TOTAL_ELEMENTS 1000000000L

int main(int argc, char **argv) {

    /* TODO 1: Initialize MPI with thread support.
     *
     * A hybrid program must tell MPI that each process will run threads.
     * Replace MPI_Init with MPI_Init_thread, requesting MPI_THREAD_FUNNELED
     * (threads may run, but only the main thread makes MPI calls):
     *
     *   int provided;
     *   MPI_Init_thread(&argc, &argv, ???, &provided);
     *
     * After getting the rank below, check that 'provided' is at least the
     * level you requested, and stop with MPI_Abort if it is not.
     */
    MPI_Init(&argc, &argv);  /* Replace this line */

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Divide the work: each rank handles a contiguous chunk */
    long my_count = TOTAL_ELEMENTS / size;
    long my_start = rank * my_count;
    /* Last rank handles any remainder */
    if (rank == size - 1)
        my_count = TOTAL_ELEMENTS - my_start;

    double t_start = MPI_Wtime();

    double partial_sum = 0.0;

    /* TODO 2: Split this rank's loop across OpenMP threads.
     *
     * Add one directive that parallelizes the loop and safely combines the
     * threads' contributions to partial_sum (see Module 3).
     */
    for (long i = my_start; i < my_start + my_count; i++) {
        partial_sum += sqrt((double)i);
    }

    double global_sum = 0.0;
    MPI_Reduce(&partial_sum, &global_sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t_start;

    if (rank == 0) {
        printf("  Ranks x threads: %d x %d\n", size, omp_get_max_threads());
        printf("  Global sum:      %.6f\n", global_sum);
        printf("  Time:            %.3f seconds\n", elapsed);
    }

    MPI_Finalize();
    return 0;
}

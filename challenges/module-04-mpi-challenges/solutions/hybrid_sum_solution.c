/*
 * hybrid_sum_solution.c -- Solution: parallel sum with MPI ranks AND
 * OpenMP threads
 *
 * From the challenge directory:
 *   mpicc -fopenmp -O2 -o hybrid_sum solutions/hybrid_sum_solution.c -lm
 *   sbatch submit_hybrid.sh
 */

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define TOTAL_ELEMENTS 1000000000L

int main(int argc, char **argv) {

    int provided;                                      /* TODO 1: SOLVED */
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long my_count = TOTAL_ELEMENTS / size;
    long my_start = rank * my_count;
    if (rank == size - 1)
        my_count = TOTAL_ELEMENTS - my_start;

    double t_start = MPI_Wtime();

    double partial_sum = 0.0;

    #pragma omp parallel for reduction(+:partial_sum)  /* TODO 2: SOLVED */
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

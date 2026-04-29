/*
 * parallel_sum_solution.c -- Solution: parallel sum with MPI_Reduce
 *
 * Compile:  mpicc -O2 -o parallel_sum parallel_sum_solution.c -lm
 * Run:      srun --ntasks=4 ./parallel_sum
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TOTAL_ELEMENTS 100000000L

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);                           /* TODO 1: SOLVED */

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);             /* TODO 2: SOLVED */
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long my_count = TOTAL_ELEMENTS / size;
    long my_start = rank * my_count;
    if (rank == size - 1)
        my_count = TOTAL_ELEMENTS - my_start;

    double t_start = MPI_Wtime();

    double partial_sum = 0.0;
    for (long i = my_start; i < my_start + my_count; i++) {
        partial_sum += sqrt((double)i);
    }

    double global_sum = 0.0;
    MPI_Reduce(&partial_sum, &global_sum, 1, MPI_DOUBLE,  /* TODO 3: SOLVED */
               MPI_SUM, 0, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - t_start;

    if (rank == 0) {
        printf("Parallel Sum Results\n");
        printf("  Ranks:       %d\n", size);
        printf("  Elements:    %ld\n", TOTAL_ELEMENTS);
        printf("  Global sum:  %.6f\n", global_sum);
        printf("  Time:        %.3f seconds\n", elapsed);
    }

    MPI_Finalize();                                    /* TODO 4: SOLVED */
    return 0;
}

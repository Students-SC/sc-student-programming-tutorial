/*
 * parallel_sum.c -- Parallel sum using MPI_Reduce
 *
 * EXERCISE: Fill in the 4 TODOs to make this program work with MPI.
 *
 * The program divides a large array across ranks. Each rank sums its
 * portion, then MPI_Reduce combines the partial sums on rank 0.
 *
 * Compile:  mpicc -O2 -o parallel_sum parallel_sum.c -lm
 * Run:      srun --ntasks=4 ./parallel_sum
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define TOTAL_ELEMENTS 100000000L

int main(int argc, char **argv) {

    /* TODO 1: Initialize MPI.
     *
     *   MPI_Init(???, ???);
     */

    /* TODO 2: Get the rank (my ID) and size (total number of ranks).
     *
     *   int rank, size;
     *   MPI_Comm_rank(???, ???);
     *   MPI_Comm_size(???, ???);
     */
    int rank = 0, size = 1;  /* Replace with real MPI calls */

    /* Divide the work: each rank handles a contiguous chunk */
    long my_count = TOTAL_ELEMENTS / size;
    long my_start = rank * my_count;
    /* Last rank handles any remainder */
    if (rank == size - 1)
        my_count = TOTAL_ELEMENTS - my_start;

    /* Each rank computes a partial sum: sum of sqrt(i) for its chunk */
    double t_start = MPI_Wtime();

    double partial_sum = 0.0;
    for (long i = my_start; i < my_start + my_count; i++) {
        partial_sum += sqrt((double)i);
    }

    /* TODO 3: Use MPI_Reduce to sum all partial sums into global_sum on rank 0.
     *
     *   MPI_Reduce(&partial_sum, &global_sum, 1, MPI_DOUBLE,
     *              ???, ???, MPI_COMM_WORLD);
     *
     * Arguments to fill in:
     *   - The reduction operation (MPI_SUM)
     *   - The root rank (0)
     */
    double global_sum = 0.0;

    double elapsed = MPI_Wtime() - t_start;

    if (rank == 0) {
        printf("Parallel Sum Results\n");
        printf("  Ranks:       %d\n", size);
        printf("  Elements:    %ld\n", TOTAL_ELEMENTS);
        printf("  Global sum:  %.6f\n", global_sum);
        printf("  Time:        %.3f seconds\n", elapsed);
    }

    /* TODO 4: Finalize MPI.
     *
     *   MPI_Finalize();
     */

    return 0;
}

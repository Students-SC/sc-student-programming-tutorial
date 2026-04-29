/*
 * mpi_hello.c -- MPI Hello World
 *
 * Each rank prints its ID and the total number of ranks.
 *
 * Compile:  mpicc -O2 -o mpi_hello mpi_hello.c
 * Run:      srun --ntasks=4 ./mpi_hello
 */

#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    printf("Hello from rank %2d of %d on %s\n", rank, size, hostname);

    MPI_Finalize();
    return 0;
}

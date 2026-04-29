/*
 * ring_solution.c -- Solution: ring communication with MPI_Sendrecv
 *
 * Compile:  mpicc -O2 -o ring ring_solution.c
 * Run:      srun --ntasks=4 ./ring
 */

#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_val = rank;
    int recv_val = -1;

    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    MPI_Sendrecv(&send_val, 1, MPI_INT, dest,   0,
                 &recv_val, 1, MPI_INT, source, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Rank %d: sent %d to rank %d, received %d from rank %d\n",
           rank, send_val, dest, recv_val, source);

    if (rank == 0) {
        if (recv_val == size - 1) {
            printf("\nRing communication successful! "
                   "Rank 0 received %d from rank %d.\n", recv_val, size - 1);
        } else {
            printf("\nRing communication FAILED. "
                   "Expected %d but got %d.\n", size - 1, recv_val);
        }
    }

    MPI_Finalize();
    return 0;
}

/*
 * ring.c -- Ring communication pattern
 *
 * CHALLENGE: Each rank sends its rank number to the next rank in a ring.
 * After one full pass, rank 0 should have received the value from the
 * last rank (size - 1).
 *
 * Compile:  mpicc -O2 -o ring ring.c
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

    /* TODO: Send send_val to 'dest' and receive into recv_val from 'source'.
     *
     * Be careful about ordering! If every rank calls MPI_Send first,
     * they'll all block waiting for someone to call MPI_Recv -- that's
     * a DEADLOCK.
     *
     * Strategy: Even-ranked processes send first, then receive.
     *           Odd-ranked processes receive first, then send.
     *
     * Alternatively, use MPI_Sendrecv which does both at once safely:
     *   MPI_Sendrecv(&send_val, 1, MPI_INT, dest,   0,
     *                &recv_val, 1, MPI_INT, source, 0,
     *                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
     *
     * YOUR CODE HERE:
     */

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

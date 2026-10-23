/*
 * ring.c -- Ring communication pattern
 *
 * CHALLENGE: Each rank sends its rank number to the next rank in a ring.
 * Each value moves one hop. Rank 0 receives the rank number from the
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
     * Be careful about ordering! MPI_Send may wait for a matching receive.
     * If every rank sends first, the program can deadlock. Do not rely on
     * message buffering to make that ordering work.
     *
     * Strategy: Even-ranked processes send first, then receive.
     *           Odd-ranked processes receive first, then send.
     *
     * Alternatively, explore MPI_Sendrecv to combine the send and receive
     * in one call. Use the documentation to work out its arguments.
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

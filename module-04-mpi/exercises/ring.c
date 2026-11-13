/*
 * ring.c -- Ring communication pattern
 *
 * EXERCISE: Each rank sends its rank number to the next rank in a ring.
 * Each value moves one hop. Rank 0 receives the rank number from the
 * last rank (size - 1).
 *
 * Compile:  mpicc -O2 -o ring ring.c
 * Run:      srun --partition=mi2101x --nodes=1 --time=1:00 --ntasks=4 timeout 10 ./ring
 * More rank counts in one job:  sbatch test_ring.sh ./ring
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

    /* Derive the expected value independently of the communication variables. */
    int expected = (rank == 0) ? size - 1 : rank - 1;
    int local_failure = (recv_val != expected);
    if (local_failure)
        fprintf(stderr, "Rank %d: expected %d but received %d\n",
                rank, expected, recv_val);

    /* Every rank participates, so one bad result makes the whole test fail. */
    int failed_ranks;
    MPI_Allreduce(&local_failure, &failed_ranks, 1, MPI_INT, MPI_SUM,
                  MPI_COMM_WORLD);

    if (rank == 0) {
        if (failed_ranks == 0)
            printf("\nRing communication successful! "
                   "All %d ranks received the expected value.\n", size);
        else
            printf("\nRing communication FAILED. "
                   "%d of %d ranks received an incorrect value.\n",
                   failed_ranks, size);
    }

    MPI_Finalize();
    return failed_ranks == 0 ? 0 : 1;
}

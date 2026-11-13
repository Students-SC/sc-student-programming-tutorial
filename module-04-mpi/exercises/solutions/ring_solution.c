/*
 * ring_solution.c -- Solution: ring communication with MPI_Sendrecv
 *
 * From the exercises directory:
 *   mpicc -O2 -o ring_solution solutions/ring_solution.c
 *   srun --partition=mi2101x --nodes=1 --time=2:00 --ntasks=4 ./ring_solution
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

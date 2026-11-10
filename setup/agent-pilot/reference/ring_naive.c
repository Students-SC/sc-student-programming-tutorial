#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send_val = rank, recv_val = -1;
    int dest = (rank + 1) % size;
    int source = (rank - 1 + size) % size;

    MPI_Send(&send_val, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
    MPI_Recv(&recv_val, 1, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Rank %d received %d\n", rank, recv_val);
    MPI_Finalize();
    return 0;
}

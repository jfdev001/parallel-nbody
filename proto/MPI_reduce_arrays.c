/*
For trying to do the following operation
P0: [0, 0, 0, 1, 2, 3]
            +
P1: [1, 2, 3, 0, 0, 0]
RESULT = [1, 2, 3, 1, 2, 3]
Bcast(RESULT)

$ mpicc -o MPI_reduce_arrays MPI_reduce_arrays.c
$ prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi ./MPI_reduce_arrays
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv) {
    // Init MPI
    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    printf("Hello %d/%d\n", rank, size);

    MPI_Finalize();
    return 0;
}
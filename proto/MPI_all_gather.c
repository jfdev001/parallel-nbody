/*
For messing with the allgather operation when each process has 
a different part of an array

mpicc -o MPI_all_gather MPI_all_gather.c
prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi ./MPI_all_gather
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MAXBODIES 100
#define BODYCT 8

void print_arr(int arr[], int rank) {
    printf("RANK %d: ", rank+1);
    for (int i = 0; i < BODYCT; i++)
        printf("%d ", arr[i]);
    printf("\n"); 
}

int main(int argc, char **argv) {


    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int positions[MAXBODIES];
    for (int position = 0; position < BODYCT; position++)
        positions[position] = 0;

    // Determine bounds for computation
    int lower_bound, upper_bound;
    int N = BODYCT;
    int P = size;
    lower_bound = rank*N/P;
    upper_bound = rank == P-1 ? (rank+1)*N/P + N%P : (rank+1)*N/P; 

    // Initialize the values of the array based on the process
    for (int b = lower_bound; b < upper_bound; b++)
        positions[b] = rank+1;

    // Print values
    print_arr(positions, rank);

    MPI_Finalize();
}
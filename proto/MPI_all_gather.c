/*
For messing with the allgather operation when each process has 
a different part of an array

mpicc -o MPI_all_gather MPI_all_gather.c
prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi ./MPI_all_gather

https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node70.html#Node70
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MAX_ELES 100
#define CUR_ELES 6

void print_arr(int arr[], int rank, int size) {
    printf("RANK %d: ", rank+1);
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
    printf("\n"); 
}

void copy_array(int *arr1, int *arr2, int arr_size){
    for (int i = 0; i < arr_size; i++) {
        arr1[i] = arr2[i];
    }
}

int main(int argc, char **argv) {


    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int positions[MAX_ELES];
    for (int position = 0; position < CUR_ELES; position++)
        positions[position] = 0;

    // Determine bounds for computation
    int lower_bound, upper_bound;
    int N = CUR_ELES;
    int P = size;
    lower_bound = rank*N/P;
    upper_bound = rank == P-1 ? (rank+1)*N/P + N%P : (rank+1)*N/P; 
    int *displacements = malloc(size*sizeof(int));
    int *rcvcounts = malloc(size*sizeof(int));
    for (int r = 0; r < P; r++) {
        displacements[r] = r*N/P;
        rcvcounts[r] = (r == P-1 ? (r+1)*N/P + N%P : (r+1)*N/P) - (r*N/P);
    }

    displacements[0] = 2;
    displacements[1] = 4;
    displacements[2] = 6;

    rcvcounts[0] = 2;
    rcvcounts[1] = 2;
    rcvcounts[2] = 2;

    if (rank == 0) {
        printf("Displacements -- ");
        print_arr(displacements, rank, size);
        printf("Rcvcounts -- ");
        print_arr(rcvcounts, rank, size);
    }

    // Initialize the values of the array based on the process
    for (int b = lower_bound; b < upper_bound; b++)
        positions[b] = rank+1;

    // Print values
    print_arr(positions, rank, CUR_ELES);

    // int gathered_positions[MAX_ELES];
    // MPI_Gather(
    //     &positions[lower_bound], upper_bound-lower_bound, MPI_INT,
    //     gathered_positions, upper_bound-lower_bound, MPI_INT, 0, comm); // receive amount might be variable

    // MPI_Gather(
    //     MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
    //     &positions[lower_bound], upper_bound-lower_bound, MPI_INT, 0, comm); 

    // MPI_Allgather(
    //     MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
    //     &positions[lower_bound], upper_bound-lower_bound, MPI_INT, comm);

    MPI_Allgatherv(
        MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        positions, rcvcounts, displacements, MPI_DOUBLE, comm);

    // if (rank == 0) {
    //     printf("---------\n");
    //     printf("RANK %d: Gathered\n", rank+1);
    //     print_arr(positions, rank);
    //     //print_arr(positions, rank);
    //     printf("---------\n");
    // }

    //MPI_Bcast(gathered_positions, CUR_ELES, MPI_INT, 0, comm);

    print_arr(positions, rank, CUR_ELES);
    //print_arr(gathered_positions, rank);



    MPI_Finalize();
}
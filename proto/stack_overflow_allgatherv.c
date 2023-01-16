/*
mpicc -o stack_overflow_allgatherv stack_overflow_allgatherv.c; prun -v -1 -np 3 -script $PRUN_ETC/prun-openmpi ./stack_overflow_allgatherv
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MAX_ELES 100
#define CUR_ELES 6

void print_arr(int arr[], int rank) {
    printf("RANK %d: ", rank+1);
    for (int i = 0; i < CUR_ELES; i++)
        printf("%d ", arr[i]);
    printf("\n"); 
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // init all elements to 0
    int array[MAX_ELES];
    for (int position = 0; position < CUR_ELES; position++)
        array[position] = 0;

    // Determine displacemenst and recv counts for data parallelism
    int lower_bound, upper_bound;
    int N = CUR_ELES;
    int P = size;
    
    int *recvcounts = malloc(size * sizeof(int));
    int *displacements = malloc(size * sizeof(int));

    int *lower_bounds = malloc(size * sizeof(int));
    int *upper_bounds = malloc(size * sizeof(int));

    for (int r = 0; r < size; r++) { 
        lower_bounds[r] = r*(N/P); 
        upper_bounds[r] = r != P-1 ? (r+1)*(N/P) : N;
        displacements[r] = lower_bounds[r];
        recvcounts[r] = upper_bounds[r] - lower_bounds[r];
    }

    lower_bound = lower_bounds[rank];
    upper_bound = upper_bounds[rank];

    print_arr(displacements, rank);
    print_arr(recvcounts, rank);

    // Initialize the values of the array based on the process rank
    for (int b = lower_bound; b < upper_bound; b++)
        array[b] = rank+1;

    // inspect array on process
    print_arr(array, rank);

    // Using all gather v
    int rcounts[3] = {2, 2, 2};
    int displs[3] = {0, 2, 4};
    int gathered_array[MAX_ELES];

    MPI_Allgatherv(
        &array[lower_bound], upper_bound-lower_bound, MPI_DOUBLE, 
        gathered_array, rcounts, displs, MPI_DOUBLE, comm);

    print_arr(gathered_array, rank);

    MPI_Finalize();
}
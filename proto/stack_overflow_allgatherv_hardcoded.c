/*
mpicc -o stack_overflow_allgatherv_hardcoded stack_overflow_allgatherv_hardcoded.c; prun -v -1 -np 3 -script $PRUN_ETC/prun-openmpi ./stack_overflow_allgatherv_hardcoded
*/

#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define MAX_ELES 100
#define CUR_ELES 6

void print_arr(int arr[], int rank) {
    printf("RANK %d: ", rank);
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

    int lower_bounds[3] = {0, 2, 4};
    int upper_bounds[3] = {2, 4, 6};
    
    lower_bound = lower_bounds[rank];
    upper_bound = upper_bounds[rank];
    printf("RANK %d: (%d,%d)\n", rank, lower_bound, upper_bound);

    // Initialize the values of the array based on the process rank
    for (int b = lower_bound; b < upper_bound; b++)
        array[b] = rank+1;

    printf("RANK %d: (%d,%d)\n", rank, lower_bound, upper_bound);

    // inspect array on process
    print_arr(array, rank);

    // Using all gather v
    int rcounts[3] = {2, 2, 2};
    int displs[3] = {0, 2, 4};

    int gathered_array[MAX_ELES];
    for (int i = 0; i < CUR_ELES; i++) { gathered_array[i] = -1;}
    

    printf("RANK %d: %d\n", rank, upper_bound-lower_bound);
    printf("RANK %d, array[lower_bound]=%d\n", rank, array[lower_bound]);
    printf("RANK %d, array[upper_bound]=%d\n", rank, array[upper_bound]);
    MPI_Gather(
        &array[lower_bound], upper_bound-lower_bound, MPI_DOUBLE, 
        gathered_array, 2, 
        MPI_DOUBLE, 0, comm);
    MPI_Bcast(gathered_array, CUR_ELES, MPI_DOUBLE, 0, comm);

    print_arr(gathered_array, rank);

    MPI_Finalize();
}
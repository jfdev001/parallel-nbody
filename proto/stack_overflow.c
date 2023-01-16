/*
mpicc -o stack_overflow stack_overflow.c; prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi ./stack_overflow
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

    // Determine allgatherv displacements and bounds
    int lower_bound, upper_bound;
    int N = CUR_ELES;
    int P = size;

    int *displs = malloc(size*sizeof(int));
    int *recvcounts = malloc(size*sizeof(int));
    int *lower_bounds = malloc(size*sizeof(int));
    int *upper_bounds = malloc(size*sizeof(int));

    for (int r = 0; r < size; r++) { 
        lower_bounds[r] = r*(N/P); 
        upper_bounds[r] = r == P-1 ? N : (r+1)*(N/P) ;
        displs[r] = lower_bounds[r];
        recvcounts[r] = upper_bounds[r] - lower_bounds[r];
    }

    lower_bound = lower_bounds[rank]; // {0, 2, 4}[rank]
    upper_bound = upper_bounds[rank]; // {2, 4, 6}[rank]

    // Initialize the values of the array based on the process rank
    for (int b = lower_bound; b < upper_bound; b++)
        array[b] = rank+1;

    // inspect array on process
    print_arr(array, rank);

    // GATHER
    int gathered_array[MAX_ELES];
    MPI_Allgatherv(
        &array[lower_bound], upper_bound-lower_bound, MPI_INT,
        // MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
        gathered_array, recvcounts, displs, MPI_INT, comm);

    // inspect gathered array on only one process, though it 
    // it present on all processes
    if (rank == 0) {
        printf("---------\n");
        printf("RANK %d: Gathered\n", rank);
        print_arr(gathered_array, rank);
        printf("---------\n");
    }

    // BCAST
    // MPI_Bcast(gathered_array, CUR_ELES, MPI_INT, 0, comm);

    MPI_Finalize();
}
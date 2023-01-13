/*
mpicc -o stack_overflow stack_overflow.c
prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi ./stack_overflow
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

    // Determine indices for data parallelism
    int lower_bound, upper_bound;
    int N = CUR_ELES;
    int P = size;
    lower_bound = rank*N/P;
    upper_bound = rank == P-1 ? (rank+1)*N/P + N%P : (rank+1)*N/P; 

    // Initialize the values of the array based on the process rank
    for (int b = lower_bound; b < upper_bound; b++)
        array[b] = rank+1;

    // inspect array on process
    print_arr(array, rank);

    // GATHER
    int gathered_array[MAX_ELES];
    MPI_Gather(
        &array[lower_bound], upper_bound-lower_bound, MPI_INT,
        gathered_array, upper_bound-lower_bound, MPI_INT, 0, comm);

    // inspect gathered array
    if (rank == 0) {
        printf("---------\n");
        printf("RANK %d: Gathered\n", rank+1);
        print_arr(gathered_array, rank);
        printf("---------\n");
    }

    // BCAST
    MPI_Bcast(gathered_array, CUR_ELES, MPI_INT, 0, comm);

    MPI_Finalize();
}
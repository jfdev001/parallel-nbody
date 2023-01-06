// Course     : Parallel Programming Practical
// Course Code: X_400162
// Professor  : Dr. H.E. Bal
// Due        : 2023-02-03
// Author     : Jared G. Frazier, 2795544
// Description: Parallel n-body simulation
  
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

void preprocess() {

}

void simulate() {

}

void postprocess() {

}

int main(int argc, char **argv) {
    // setup mpi
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    

    // Print hello
    printf("Hello from %d/%d\n", rank, size);

    // tear down mpi
    MPI_Finalize();
}

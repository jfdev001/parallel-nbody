/* 
Course     : Parallel Programming Practical
Course Code: X_400162
Professor  : Dr. H.E. Bal
Due        : 2023-02-03
Author     : Jared G. Frazier, 2795544
Description: Parallel n-body simulation
Example Cmd: 
$ # Requires `module load openmpi/gcc`
$ prun -np 2 -script $PRUN_ETC/prun-openmpi ./nbody/nbody-par 
Notes:
* What data needs to be scattered?
    - What MPI datatypes might be used here?
* What are the data dependencies of different functions?
    - Could latency hiding (non-blocking calls) be used?
    - Do ghost cells need to be used?
    - How can I debug outputs?
*/ 
  
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

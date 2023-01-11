/*
For testing struct reduction operations with MPI
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


#define MAXBODIES 10000

struct body {
    double value;
};

struct world {
    struct body bodies[MAXBODIES];
};

void build_mpi_body_type(){
    return ;
}

/// @brief Any need to do this? Can i just MPI_ALlreduce on the MPI bodies only
/// @details E.g., Allreduce(MPI_IN_PLACE, world.bodies, OPERATION, ...)
void build_mpi_world_type(){
    return ;
}

void mpi_struct_op(){
    return ;
}

int main(int argc, char **argv) {

    // Init mpi stuff
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);

    // Build the mpi body type
    build_mpi_body_type();
    build_mpi_world_type();

    // Init data

    
    MPI_Finalize();
}
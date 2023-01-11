/*
For testing struct reduction operations with MPI.

References:
https://stackoverflow.com/questions/66622459/sending-array-of-structs-in-mpi
https://stackoverflow.com/questions/22709015/mpi-derived-type-send/22714292?noredirect=1#22714292
https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node80.html
https://stackoverflow.com/questions/29184378/mpi-allreduce-on-an-array-inside-a-dynamic-array-of-structure

Example:
mpicc -o MPI_reduce_struct MPI_reduce_struct.c
prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi ./MPI_reduce_struct
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define MAXBODIES 10000
#define BODYCT 10

struct body {
    double xf;
    double yf;
};

struct world {
    struct body bodies[MAXBODIES];
    int bodyCt;
};

void build_mpi_body_type(MPI_Datatype *MPI_BODY_TYPE){
    // n members
    const int count = 2;

    // length of a given members data
    int block_lengths[count] = {1, 1};

    // data types of members
    MPI_Datatype types[count] = {MPI_DOUBLE, MPI_DOUBLE};

    // displacements of memebrs
    struct body body;
    MPI_Aint displacements[count];
    MPI_Aint base, member_offset;

    MPI_Get_address(&body, &base);
    MPI_Get_address(&body.xf, &member_offset);
    displacements[1] = member_offset - base; 

    MPI_Get_address(&body, &base);
    MPI_Get_address(&body.yf, &member_offset);
    displacements[2] = member_offset - base;

    // create type of member
    MPI_Type_create_struct(
        count,
        block_lengths,
        displacements,
        types,
        MPI_BODY_TYPE);

    MPI_Type_commit(MPI_BODY_TYPE);

    return;
}

/// @brief Any need to do this? Can i just MPI_ALlreduce on the MPI bodies only
/// @details E.g., Allreduce(MPI_IN_PLACE, world.bodies, OPERATION, ...)
void build_mpi_world_type(){
    return;
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
    // build_mpi_body_type();
    // build_mpi_world_type();

    // Init data
    struct world *world = calloc(1, sizeof(*world));
    world->bodyCt = BODYCT;
    for (int b = 0; b < world->bodyCt; b++) {
        world->bodies[b].xf = 1; 
        world->bodies[b].yf = 2;
    }
    
    MPI_Finalize();
}
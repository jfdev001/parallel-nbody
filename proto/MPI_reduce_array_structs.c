/*
For testing struct reduction operations with MPI.

References:
https://stackoverflow.com/questions/66622459/sending-array-of-structs-in-mpi
https://stackoverflow.com/questions/22709015/mpi-derived-type-send/22714292?noredirect=1#22714292
https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node80.html
https://stackoverflow.com/questions/29184378/mpi-allreduce-on-an-array-inside-a-dynamic-array-of-structure

Closest thing to what i'm trying to do here.
[Stack overflow: MPI Struct of Structs](https://stackoverflow.com/questions/5127049/mpi-send-and-receive-struct-of-structs)

Struct of Structs w.r.t Particles? DOUBLE CHECK THIS!!!
[Supercomputing and Apps Devlopment Presentation](https://materials.prace-ri.eu/386/1/MPI2-DataTypes_2014.pdf)

[Openmpi forum: Struct Creation](https://users.open-mpi.narkive.com/ZmgS4RgF/ompi-mpi-type-struct-for-structs-with-dynamic-arrays)



Quinn 2003, Apapendix C on debugging (fflush)

Example:
mpicc -o MPI_reduce_array_structs MPI_reduce_array_structs.c
prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi ./MPI_reduce_array_structs
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

#define MAXBODIES 1000
#define BODYCT 10

// MPI datatypes and operations
// inspired by [Github: Ch. 8 of Manning Parallel and HPC, 2021](https://github.com/essentialsofparallelcomputing/Chapter8/blob/12f16453c4995b6090192d97cc128d798ff435de/GlobalSums/globalsums.c])
MPI_Datatype BODY_TYPE;
MPI_Datatype N_BODIES_TYPE;
MPI_Datatype WORLD_TYPE;
MPI_Op SUM_FORCES;

struct body {
    double xf;
    double yf;
};

struct world {
    int bodyCt;
    struct body bodies[MAXBODIES]; // it seems like a contiguous type will be needed for this 
};

/// @brief Define a custom MPI type for the celestial body
void build_mpi_body_type(){
    // n members
    //const int count = 2;

    // length of a given members data
    int block_lengths[2] = {1, 1};

    // data types of members
    MPI_Datatype types[2] = {MPI_DOUBLE, MPI_DOUBLE};

    // displacements of memebrs
    struct body body;
    MPI_Aint displacements[2];
    MPI_Aint base, member_offset;

    MPI_Get_address(&body, &base);
    MPI_Get_address(&body.xf, &member_offset);
    displacements[1] = member_offset - base; 
    printf("displacements[1] = %td\n", displacements[1]);

    MPI_Get_address(&body.yf, &member_offset);
    displacements[2] = member_offset - base;
    printf("displacements[2] = %td\n", displacements[2]);

    // create type and commit
    MPI_Type_create_struct(
        2,
        block_lengths,
        displacements,
        types,
        &BODY_TYPE);

    MPI_Type_commit(&BODY_TYPE);

    return;
}

/// @brief Creates a continous type for the N-bodies
/// @details Inspiration for this comes from
/// [Stack Overflow: Struct of Structs](https://stackoverflow.com/questions/5127049/mpi-send-and-receive-struct-of-structs)
/// and p.272 of Parallel and HPC (2021, Manning)
void build_mpi_n_bodies_type(){
    MPI_Type_contiguous(MAXBODIES, BODY_TYPE, &N_BODIES_TYPE);
    MPI_Type_commit(&N_BODIES_TYPE);
    return;
}

/// @brief Define a custom MPI datatype for the simulation world
/// @details Is this necessary? I am just going to modify the bodies
/// member of the struct world directly.... it's not like i'm passing
/// the struct world to the other processes..
/// Perhaps an MPI contiguous type is needed here?
/// https://stackoverflow.com/questions/5127049/mpi-send-and-receive-struct-of-structs
void build_mpi_world_type(){

    // members, elements in members, and member datatypes
    //const int count = 2;
    int block_lengths[2] = {1, 1};
    printf("completed `block_lengths[2]`\n"); fflush(stdout);

    MPI_Datatype types[2] = {MPI_INT, BODY_TYPE};
    printf("completed `types[2]`\n"); fflush(stdout);

    /*compute displacements
    */
    MPI_Aint displacements[1];
    printf("completed `displacements[2]`\n"); fflush(stdout);

    // base address for struct
    MPI_Aint base;
    struct world world;
    MPI_Get_address(&world, &base);
    printf("MPI_Get_address(&world, &base)\n"); fflush(stdout);

    // member addresses
    MPI_Aint member_offset;

    MPI_Get_address(&world.bodyCt, &member_offset);
    displacements[1] = member_offset - base;
    printf("displacements[1] = member_offset - base;\n"); fflush(stdout);
    printf("displacements[1] = %td == %x\n", displacements[1], displacements[1]);

    MPI_Get_address(&world.bodies[0], &member_offset);
    displacements[2] = member_offset - base;
    printf("displacements[2] = member_offset - base;\n"); fflush(stdout);
    printf("displacements[2] = %td == %x\n", displacements[2], displacements[2]);
    

    // Create datatype
    MPI_Type_create_struct(
        2,
        block_lengths,
        displacements,
        types,
        &WORLD_TYPE);

     printf(
        "MPI_Type_create_struct(2,block_lengths,displacements,types,&MPI_WORLD_TYPE);\n"); fflush(stdout);

    MPI_Type_commit(&WORLD_TYPE);
    printf("MPI_Type_commit(&MPI_WORLD_TYPE);\n"); fflush(stdout);

    return;
}

/// @brief Sums X and Y forces of a bodies array
/// @details Must match the MPI_User_function parameters
/// see [Open-mpi docs: MPI_Op_create](https://www.open-mpi.org/doc/v3.0/man3/MPI_Op_create.3.php)
/// The ref @ [Github: Ch. 8 of Manning Parallel and HPC, 2021](https://github.com/essentialsofparallelcomputing/Chapter8/blob/12f16453c4995b6090192d97cc128d798ff435de/GlobalSums/globalsums.c])
/// appears to directly use the struct pointer type
/// [Allreduce on array of structs](https://stackoverflow.com/questions/29184378/mpi-allreduce-on-an-array-inside-a-dynamic-array-of-structure)
/// What is the length here? Documentation suggests that the length of the buffer
/// (which presumably is determined by the datatype itself @ref `build_mpi_body_type`)
void sum_forces(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){

    // Reducing the simulation world is probably the best option
    // because otherwise the length of the array bodies array will
    // be MAX ele (which is presumably far less than the bodyCt)
    struct world *in_world = invec;
    struct world *inout_world = inoutvec;

    // Sum the forces 
    for (int b = 0; b < in_world->bodyCt; b++){
        (inout_world)->bodies[b].xf += (in_world)->bodies[b].xf;
        (inout_world)->bodies[b].yf += (in_world)->bodies[b].yf;
    }

    return;
}

void print_world(struct world *world){
    for (int b = 0; b < world->bodyCt; b++){
        printf("%5.3f %5.3f\n", (world)->bodies[b].xf, (world)->bodies[b].yf);
    }
}


int main(int argc, char **argv) {

    // Init mpi stuff
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int size, rank;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);


    // Build the mpi types
    build_mpi_body_type();
    printf("RANK %d: completed build_mpi_body_type\n", rank); fflush(stdout);

    build_mpi_n_bodies_type();
    printf("RANK %d: completed build_mpi_n_bodies_type()\n", rank); fflush(stdout);

    // build_mpi_world_type();
    // printf("RANK %d: completed build_mpi_world_type\n", rank); fflush(stdout);

    MPI_Op_create((MPI_User_function *)sum_forces, true, &SUM_FORCES);
    printf("RANK %d: completed MPI_Op_create\n", rank); fflush(stdout);

    // Init data
    struct world *world = calloc(1, sizeof(*world));
    world->bodyCt = BODYCT;
    for (int b = 0; b < world->bodyCt; b++) {
        world->bodies[b].xf = 1; 
        world->bodies[b].yf = 2;
    }

    // Perform reduce operation
    // Expected results for forces should be any given xf == 2
    // and any given yf == 4
    printf("RANK %d: Perform Summation", rank);
    fflush(stdout);
    
    MPI_Finalize();
}
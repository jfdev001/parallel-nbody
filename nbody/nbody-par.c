/*
Course     : Parallel Programming Practical
Course Code: X_400162
Professor  : Dr. H.E. Bal
Due        : 2023-02-03
Author     : Jared G. Frazier, 2795544
Description: Parallel n-body simulation
Example Cmd:
$ # Requires `module load openmpi/gcc`
$ # Expects the same args as the serial script... though add an additional arg for func to print
$ prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi ./nbody-par

Notes:
* What data needs to be scattered?
    - What MPI datatypes might be used here?
    - Think about the actual parallelization of this ...
* What are the data dependencies of different functions?
    - Could latency hiding (non-blocking calls) be used?
    - Do ghost cells need to be used?
    - How can I debug outputs?


For the purpose of handling start/end indices consider offsets of ghost cells
for example
if P_0 gets bodies 1-3  &local_bodies[0] <-- 0 1 2
if P_1 gets bodies 4-6  &local_bodies[3] <-- x x x 3 4 5 x x x
if P_2 gets bodies 7-9  &local_bodies[6] <-- x x x x x x 6 7 8
therefore, there is an offset for accessing the local bodies and interproc
comm can simply be to the appropriate cells 

The indices for b and c start and stop for the computable portions of the 
the bodies (i.e., what the process actually has access to) is 
b in range(rank*n/p, (rank+1)*n/p)
c in range(rank*n/p + 1, (rank+1)*n/p) <--- could do lazy if rank == last, n/p+n%p
offset = rank*n/p

If i split up the data... then i end up requiring communication between
all processes because all processes must update forces
based on other bodies 

Repeat calculations should be fine. Consider the following situation:
P = 2, N = 6
P_0 = [0, 1, 2, x, x, x]
P_1 = [x, x, x, 3, 4, 5]
compute_forces(P_1, P_0) --> 3 -> 0, 1, 2; 4 -> 0, 1, 2; 5 -> 0, 1, 2
compute_forces(P_0, P_1) --> 0 -> 3, 4, 5; 1 -> 3, 4, 5; 2 -> 3, 4, 5
These overlap
P_1: (3, 0); (3, 1); (3, 2); (4, 0)
P_0: (0, 3);                 (0, 4) ... etc.
The alternative would be increasing the communication overhead and 
load balancing computations 

For the P = 2, N = 6 case, the following communication would occur:
```
data = scatter(world.bodies)
P0: send(&data[rank*n/p],    n/p, P1)  # 
P1: recv(&data[(rank-1)*n/p, n/p, P0]) # [x->0, x->1, x->2, 3, 4, 5]
```

How independent can these updates be? Consider the following:
P0: b = 0, 1... 
P1: b = 2, 3

Pseudocode:

struct bodyType bodies[MAX]; // big allocation on stack for all procs

if root 
    initialize(world)

struct bodyType local_bodies[MAX] = scatterv(world.bodies)
struct world *local_world = calloc(1, sizeof(*local_world)) // why is this malloced?

local_world.bodies = local_bodies 

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <mpi.h>


#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015


struct bodyType {
    double x[2];        /* Old and new X-axis coordinates */
    double y[2];        /* Old and new Y-axis coordinates */
    double xf;          /* force along X-axis */
    double yf;          /* force along Y-axis */
    double xv;          /* velocity along X-axis */
    double yv;          /* velocity along Y-axis */
    double mass;        /* Mass of the body */
    double radius;      /* width (derived from mass) */
};


struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};

/* MPI Datatypes
*/

/// @brief Build the MPI datatype for `struct bodyType`
/// @details 
/// Pacheco 1997, p. 93
/// [Example Code](https://rookiehpc.github.io/mpi/docs/mpi_type_create_struct/index.html)
/// [Stack Overflow: Sending Array of Structs](https://stackoverflow.com/questions/66622459/sending-array-of-structs-in-mpi)
static void 
build_mpi_body_type(
    double x[2],        
    double y[2],        
    double *xf,          
    double *yf,          
    double *xv,          
    double *yv,        
    double *mass,       
    double *radius,
    MPI_Datatype *MPI_Body_type) 
{
    // Number of members in struct
    const int n_members = 8;

    // Number of elements in each member
    int member_lengths[n_members];
    member_lengths[0] = 2;
    member_lengths[1] = 2;
    for (int member = 2; member < n_members; member++)
        member_lengths[member] = 1;

    // Types of members
    MPI_Datatype typearr[n_members];
    for (int member = 0; member < n_members; member++)
        typearr[member] = MPI_DOUBLE;

    // Displacements of members in memory
    MPI_Aint displacements[n_members];
    MPI_Aint start_address, address;

    displacements[0] = 0; // first member is at displacement 0
    MPI_Address(x, &start_address);

    MPI_Address(y, &address);
    displacements[1] = address - start_address;

    MPI_Address(xf, &address);
    displacements[2] = address - start_address;

    MPI_Address(yf, &address);
    displacements[3] = address - start_address;

    MPI_Address(xv, &address);
    displacements[4] = address - start_address;

    MPI_Address(yv, &address);
    displacements[5] = address - start_address;

    MPI_Address(mass, &address);
    displacements[6] = address - start_address;

    MPI_Address(radius, &address);
    displacements[7] = address - start_address;

    // Build derived datatype
    MPI_Type_create_struct( 
        n_members, 
        member_lengths,
        displacements,
        typearr,
        MPI_Body_type);

    // Register the datatype
    MPI_Type_commit(MPI_Body_type);

    return;
}

/// @brief Build the MPI datatype for `struct world`
static void 
build_mpi_world_type(
    struct bodyType bodies[MAXBODIES],
    int *bodyCt,
    int *old,
    int *xdim,
    int *ydim,
    MPI_Datatype *MPI_World_type) 
{

}


/*  Macros to hide memory layout
*/
#define X(w, B)        (w)->bodies[B].x[(w)->old]
#define XN(w, B)       (w)->bodies[B].x[(w)->old^1]
#define Y(w, B)        (w)->bodies[B].y[(w)->old]
#define YN(w, B)       (w)->bodies[B].y[(w)->old^1]
#define XF(w, B)       (w)->bodies[B].xf
#define YF(w, B)       (w)->bodies[B].yf
#define XV(w, B)       (w)->bodies[B].xv
#define YV(w, B)       (w)->bodies[B].yv
#define R(w, B)        (w)->bodies[B].radius
#define M(w, B)        (w)->bodies[B].mass

static void
clear_forces(struct world *world)
{
    int b;

    /* Clear force accumulation variables */
    for (b = 0; b < world->bodyCt; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}

static void
compute_forces(struct world *world)
{
    int b, c;

    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)....
       if you use the t+1 update then you are using the wrong forces
    */
    for (b = 0; b < world->bodyCt; ++b) {
        for (c = b + 1; c < world->bodyCt; ++c) {
            double dx = X(world, c) - X(world, b);
            double dy = Y(world, c) - Y(world, b);
            double angle = atan2(dy, dx);
            double dsqr = dx*dx + dy*dy;
            double mindist = R(world, b) + R(world, c);
            double mindsqr = mindist*mindist;
            double forced = ((dsqr < mindsqr) ? mindsqr : dsqr);
            double force = M(world, b) * M(world, c) * GRAVITY / forced;
            double xf = force * cos(angle);
            double yf = force * sin(angle);

            /* Slightly sneaky...
               force of b on c is negative of c on b;
            */
            XF(world, b) += xf;
            YF(world, b) += yf;
            XF(world, c) -= xf;
            YF(world, c) -= yf;
        }
    }
}

static void
compute_velocities(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        double xv = XV(world, b);
        double yv = YV(world, b);
        double force = sqrt(xv*xv + yv*yv) * FRICTION;
        double angle = atan2(yv, xv);
        double xf = XF(world, b) - (force * cos(angle));
        double yf = YF(world, b) - (force * sin(angle));

        XV(world, b) += (xf / M(world, b)) * DELTA_T;
        YV(world, b) += (yf / M(world, b)) * DELTA_T;
    }
}

static void
compute_positions(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        double xn = X(world, b) + XV(world, b) * DELTA_T;
        double yn = Y(world, b) + YV(world, b) * DELTA_T;

        /* Bounce off image "walls" */
        if (xn < 0) {
            xn = 0;
            XV(world, b) = -XV(world, b);
        } else if (xn >= world->xdim) {
            xn = world->xdim - 1;
            XV(world, b) = -XV(world, b);
        }
        if (yn < 0) {
            yn = 0;
            YV(world, b) = -YV(world, b);
        } else if (yn >= world->ydim) {
            yn = world->ydim - 1;
            YV(world, b) = -YV(world, b);
        }

        /* Update position */
        XN(world, b) = xn;
        YN(world, b) = yn;
    }
}

void preprocess()
{
}

void simulate()
{
}

void postprocess()
{
}

int main(int argc, char **argv)
{
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

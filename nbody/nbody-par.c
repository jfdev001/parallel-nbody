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
$ prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi nbody/nbody-par 8 0 nbody.ppm 1000
$ prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi nbody/nbody-par 4 0 nbody.ppm 1

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
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>
#include <mpi.h>

/* Constants
*/
#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015

#define NUM_BODY_TYPE_MEMBERS 8
#define NUM_WORLD_TYPE_MEMBERS 5

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

/* data structures
*/

/// @brief A body in an N-body simulation
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

/// @brief The simulation world 
struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};

/* MPI types and op
*/
MPI_Datatype BODY_TYPE;
MPI_Datatype WORLD_TYPE;
MPI_Op SUM_FORCES;


/// @brief Build the MPI datatype for `struct bodyType`
/// @details 
/// Pacheco 1997, p. 93
/// [Example Code](https://rookiehpc.github.io/mpi/docs/mpi_type_create_struct/index.html)
/// [Stack Overflow: Sending Array of Structs](https://stackoverflow.com/questions/66622459/sending-array-of-structs-in-mpi)
static void 
build_mpi_body_type() 
{
    // The struct itself
    struct bodyType body;

    // Number of elements in each member
    int member_lengths[NUM_BODY_TYPE_MEMBERS];
    member_lengths[0] = 2;
    member_lengths[1] = 2;
    for (int member = 2; member < NUM_BODY_TYPE_MEMBERS; member++)
        member_lengths[member] = 1;

    // Types of members
    MPI_Datatype typearr[NUM_BODY_TYPE_MEMBERS];
    for (int member = 0; member < NUM_BODY_TYPE_MEMBERS; member++)
        typearr[member] = MPI_DOUBLE;

    // Displacements of members in memory
    MPI_Aint displacements[NUM_BODY_TYPE_MEMBERS];
    MPI_Aint start_address, address;
    MPI_Get_address(&body, &start_address);

    MPI_Get_address(body.x, &address);
    displacements[0] = address - start_address;

    MPI_Get_address(body.y, &address);
    displacements[1] = address - start_address;

    MPI_Get_address(&body.xf, &address);
    displacements[2] = address - start_address;

    MPI_Get_address(&body.yf, &address);
    displacements[3] = address - start_address;

    MPI_Get_address(&body.xv, &address);
    displacements[4] = address - start_address;

    MPI_Get_address(&body.yv, &address);
    displacements[5] = address - start_address;

    MPI_Get_address(&body.mass, &address);
    displacements[6] = address - start_address;

    MPI_Get_address(&body.radius, &address);
    displacements[7] = address - start_address;

    // Build derived datatype
    MPI_Type_create_struct( 
        NUM_BODY_TYPE_MEMBERS, 
        member_lengths,
        displacements,
        typearr,
        &BODY_TYPE);

    // Register the datatype
    MPI_Type_commit(&BODY_TYPE);

    return;
}

/// @brief Build the MPI datatype for `struct world`
/// @details Is this needed?
static void 
build_mpi_world_type() 
{
    struct world world;
    int member_lengths[NUM_WORLD_TYPE_MEMBERS] =   {MAXBODIES, 1,       1,       1,       1};
    MPI_Datatype typearr[NUM_WORLD_TYPE_MEMBERS] = {BODY_TYPE, MPI_INT, MPI_INT, MPI_INT, MPI_INT};

    MPI_Aint displacements[NUM_WORLD_TYPE_MEMBERS];
    MPI_Aint start_address, address;

    MPI_Get_address(&world, &start_address);

    MPI_Get_address(&world.bodies[0], &address);
    displacements[0] = address - start_address;

    MPI_Get_address(&world.bodyCt, &address);
    displacements[1] = address - start_address;

    MPI_Get_address(&world.old, &address);
    displacements[2] = address - start_address;

    MPI_Get_address(&world.xdim, &address);
    displacements[3] = address - start_address;

    MPI_Get_address(&world.ydim, &address);
    displacements[4] = address - start_address;

    MPI_Type_create_struct(
        NUM_WORLD_TYPE_MEMBERS,
        member_lengths,
        displacements,
        typearr,
        &WORLD_TYPE);

    MPI_Type_commit(&WORLD_TYPE);
}

/* MPI Operators
*/

/// @brief Sums X and Y forces of a bodies array
/// @details Must match the MPI_User_function parameters
/// see [Open-mpi docs: MPI_Op_create](https://www.open-mpi.org/doc/v3.0/man3/MPI_Op_create.3.php)
/// The ref @ [Github: Ch. 8 of Manning Parallel and HPC, 2021](https://github.com/essentialsofparallelcomputing/Chapter8/blob/12f16453c4995b6090192d97cc128d798ff435de/GlobalSums/globalsums.c])
/// appears to directly use the struct pointer type
/// [Allreduce on array of structs](https://stackoverflow.com/questions/29184378/mpi-allreduce-on-an-array-inside-a-dynamic-array-of-structure)
void sum_forces(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype){

    // Reducing the simulation world is probably the best option
    // because otherwise the length of the array bodies array will
    // be MAX ele (which is presumably far less than the bodyCt)
    struct world *in_world = invec;
    struct world *inout_world = inoutvec;

    // Sum the forces 
    for (int b = 0; b < in_world->bodyCt; b++){
        XF(inout_world, b) += XF(in_world, b);
        YF(inout_world, b) += YF(in_world, b);
    }

    return;
}

/* Preprocessing
*/

/// @brief Initializes the simulation data in the n-body world
static void 
initialize_simulation_data(struct world *world)
{
    srand(SEED);
    for (int b = 0; b < world->bodyCt; ++b) {
        X(world, b) = (rand() % world->xdim);
        Y(world, b) = (rand() % world->ydim);
        R(world, b) = 1 + ((b*b + 1.0) * sqrt(1.0 * ((world->xdim * world->xdim) + (world->ydim * world->ydim)))) /
                (25.0 * (world->bodyCt*world->bodyCt + 1.0));
        M(world, b) = R(world, b) * R(world, b) * R(world, b);
        XV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
        YV(world, b) = ((rand() % 20000) - 10000) / 2000.0;
    }
    return;
}

/* N-body updates
*/

/// @brief Clear force accumulation variables
static void
clear_forces(struct world *world)
{
    for (int b = 0; b < world->bodyCt; ++b) {
        YF(world, b) = XF(world, b) = 0;
    }
}

/// @brief  Incrementally accumulate forces from each body pair
static void
compute_forces(struct world *world, int lower_bound, int upper_bound)
{
    for (int b = lower_bound; b < upper_bound; ++b) {
        for (int c = b + 1; c < world->bodyCt; ++c) {
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

/// @brief Compute using body members
static void
compute_velocities(struct world *world, int lower_bound, int upper_bound)
{

    for (int b = lower_bound; b < upper_bound; ++b) {
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

/// @brief Compute new positions of bodies
static void
compute_positions(struct world *world, int lower_bound, int upper_bound)
{
    for (int b = lower_bound; b < upper_bound; ++b) {
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


/*  Graphic output stuff...
 */

struct filemap {
    int            fd;
    off_t          fsize;
    void          *map;
    unsigned char *image;
};


static void
filemap_close(struct filemap *filemap)
{
    if (filemap->fd == -1) {
        return;
    }
    close(filemap->fd);
    if (filemap->map == MAP_FAILED) {
        return;
    }
    munmap(filemap->map, filemap->fsize);
}


static unsigned char *
Eat_Space(unsigned char *p)
{
    while ((*p == ' ') ||
           (*p == '\t') ||
           (*p == '\n') ||
           (*p == '\r') ||
           (*p == '#')) {
        if (*p == '#') {
            while (*(++p) != '\n') {
                // skip until EOL
            }
        }
        ++p;
    }

    return p;
}


static unsigned char *
Get_Number(unsigned char *p, int *n)
{
    p = Eat_Space(p);  /* Eat white space and comments */

    int charval = *p;
    if ((charval < '0') || (charval > '9')) {
        errno = EPROTO;
        return NULL;
    }

    *n = (charval - '0');
    charval = *(++p);
    while ((charval >= '0') && (charval <= '9')) {
        *n *= 10;
        *n += (charval - '0');
        charval = *(++p);
    }

    return p;
}


static int
map_P6(const char *filename, int *xdim, int *ydim, struct filemap *filemap)
{
    /* The following is a fast and sloppy way to
       map a color raw PPM (P6) image file
    */
    int maxval;
    unsigned char *p;

    /* First, open the file... */
    if ((filemap->fd = open(filename, O_RDWR)) < 0) {
        goto ppm_abort;
    }

    /* Read size and map the whole file... */
    filemap->fsize = lseek(filemap->fd, (off_t)0, SEEK_END);
    filemap->map = mmap(0,                      // Put it anywhere
                        filemap->fsize,         // Map the whole file
                        (PROT_READ|PROT_WRITE), // Read/write
                        MAP_SHARED,             // Not just for me
                        filemap->fd,            // The file
                        0);                     // Right from the start
    if (filemap->map == MAP_FAILED) {
        goto ppm_abort;
    }

    /* File should now be mapped; read magic value */
    p = filemap->map;
    if (*(p++) != 'P') goto ppm_abort;
    switch (*(p++)) {
    case '6':
        break;
    default:
        errno = EPROTO;
        goto ppm_abort;
    }

    p = Get_Number(p, xdim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, ydim);            // Get image width */
    if (p == NULL) goto ppm_abort;
    p = Get_Number(p, &maxval);         // Get image max value */
    if (p == NULL) goto ppm_abort;

    /* Should be 8-bit binary after one whitespace char... */
    if (maxval > 255) {
        goto ppm_abort;
    }
    if ((*p != ' ') &&
        (*p != '\t') &&
        (*p != '\n') &&
        (*p != '\r')) {
        errno = EPROTO;
        goto ppm_abort;
    }

    /* Here we are... next byte begins the 24-bit data */
    filemap->image = p + 1;

    return 0;

ppm_abort:
    filemap_close(filemap);

    return -1;
}


static inline void
color(const struct world *world, unsigned char *image, int x, int y, int b)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));
    int tint = ((0xfff * (b + 1)) / (world->bodyCt + 2));

    p[0] = (tint & 0xf) << 4;
    p[1] = (tint & 0xf0);
    p[2] = (tint & 0xf00) >> 4;
}

static inline void
black(const struct world *world, unsigned char *image, int x, int y)
{
    unsigned char *p = image + (3 * (x + (y * world->xdim)));

    p[2] = (p[1] = (p[0] = 0));
}

static void
display(const struct world *world, unsigned char *image)
{
    double i, j;
    int b;

    /* For each pixel */
    for (j = 0; j < world->ydim; ++j) {
        for (i = 0; i < world->xdim; ++i) {
            /* Find the first body covering here */
            for (b = 0; b < world->bodyCt; ++b) {
                double dy = Y(world, b) - j;
                double dx = X(world, b) - i;
                double d = sqrt(dx*dx + dy*dy);

                if (d <= R(world, b)+0.5) {
                    /* This is it */
                    color(world, image, i, j, b);
                    goto colored;
                }
            }

            /* No object -- empty space */
            black(world, image, i, j);

colored:        ;
        }
    }
}

/* Printing world member data
*/

/// @brief Print the forces in the world structure
static void 
print_forces(struct world *world)
{
    for (int b = 0; b < world->bodyCt; b++)
    {
        printf("%10.3f %10.3f\n", XF(world, b), YF(world, b));
    }
    return;
}

/// @brief Print the velocities in the world structure
static void 
print_velocities(struct world *world)
{
    for (int b = 0; b < world->bodyCt; b++)
    {
        printf("%10.3f %10.3f\n", XV(world, b), YV(world, b));
    }
    return;
}

/// @brief Print the positions in the world structure
static void 
print_positions(struct world *world)
{
    for (int b = 0; b < world->bodyCt; b++)
    {
        printf("%10.3f %10.3f\n", X(world, b), Y(world, b));
    }
    return;
}

static void
print(struct world *world)
{
    int b;

    for (b = 0; b < world->bodyCt; ++b) {
        printf("%10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n",
               X(world, b), Y(world, b), XF(world, b), YF(world, b), XV(world, b), YV(world, b));
        fflush(stdout);
    }
}

static long long
nr_flops(int n, int steps) {
  long long nr_flops = 0;
  // compute forces
  nr_flops += 20 * (n * (n-1) / 2);
  // compute velocities
  nr_flops += 18 * n;
  // compute positions
  nr_flops += 4 * n;

  nr_flops *= steps;

  return nr_flops;
}
    

int main(int argc, char **argv)
{
    /****************
    * Pre-processing
    *****************/

    // Setup mpi
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank;
    int size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Build MPI types and ops
    build_mpi_body_type();
    build_mpi_world_type();
    MPI_Op_create((MPI_User_function *)sum_forces, true, &SUM_FORCES);

    // Measurement variables
    unsigned int lastup = 0;
    unsigned int secsup;
    int steps;                // number of timesteps
    double rtime;
    struct timeval start;
    struct timeval end;       
    struct filemap image_map; // for graphics

    // For testing output
    bool check_forces = false;
    bool check_positions = false;
    bool check_velocities = false;
    bool check_world = false;
    bool check_performance = false;

    // Allocate nbody world
    struct world *world = calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters */
    if (argc < 5) {
        if (rank == 0) {
            fprintf(
                stderr, 
                "Usage: %s num_bodies secs_per_update ppm_output_file steps %s\n",
                argv[0],
                "[--check-forces, --check-positions, --check-velocities, --check-world, --check-performance]");
        }
        exit(1);
    }

    // Set testing flags
    for (int flag = 5; flag < argc; flag++)
    {
        if (strcmp(argv[flag], "--check-forces") == 0) 
        {
            check_forces = true;

        } else if (strcmp(argv[flag], "--check-positions") == 0)
        {
            check_positions = true;

        } else if (strcmp(argv[flag], "--check-velocities") == 0)
        {
            check_velocities = true;

        } else if (strcmp(argv[flag], "--check-world") == 0)
        {
            check_world = true;

        } else if (strcmp(argv[flag], "--check-performance") == 0)
        {
            check_performance = true;

        } else {
            if (rank == 0) { 
                printf("EXIT: Unrecognized flag `%s`\n", argv[flag]); 
            }
            exit(1);
        }
    }

    // For production
    if (argc == 5){
        check_world = true;
        check_performance = true;
    }

    // Set bodies
    if ((world->bodyCt = atol(argv[1])) > MAXBODIES ) {
        fprintf(stderr, "Using only %d bodies...\n", MAXBODIES);
        world->bodyCt = MAXBODIES;

    } else if (world->bodyCt < 2) {
        fprintf(stderr, "Using two bodies...\n");
        world->bodyCt = 2;
    }
    
    // Graphics
    secsup = atoi(argv[2]);
    if (map_P6(argv[3], &world->xdim, &world->ydim, &image_map) == -1) {
        fprintf(stderr, "Cannot read %s: %s\n", argv[3], strerror(errno));
        exit(1);
    }

    // Set timesteps
    steps = atoi(argv[4]);

    // Print meta information to stderr (shouldn't effect diff check from stdout)
    if (rank == 0) {
        fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);
    }

    // Initialize nbody world
    initialize_simulation_data(world);

    // Determine bounds for computation
    int lower_bound, upper_bound;
    int N = world->bodyCt;
    int P = size;
    lower_bound = rank*N/P;
    upper_bound = rank == P-1 ? (rank+1)*N/P + N%P : (rank+1)*N/P; // naive index soln

    printf("RANK %d: lb=%d, ub=%d\n", rank, lower_bound, upper_bound);

    /****************
    * Main processing
    *****************/

    // nbody algo here
    // NOTE: The correct positions and forces are caculated after a single iteration
    // iteration two, there appears to be a problem
    for (int step = 0; step < steps; step++) {
        clear_forces(world);

        // Force updates are local and then local forces are summed across processes
        compute_forces(world, lower_bound, upper_bound);
        MPI_Allreduce(MPI_IN_PLACE, world, 1, WORLD_TYPE, SUM_FORCES, comm); 

        compute_velocities(world, lower_bound, upper_bound);
        compute_positions(world, lower_bound, upper_bound);

        // Force updates require newest positions, therefore must do allgather
        // https://www.open-mpi.org/doc/v3.0/man3/MPI_Allgather.3.php

        // MPI_Allgather(
        //     &world->bodies[lower_bound], upper_bound-lower_bound, BODY_TYPE,
        //     &world->bodies[upper_bound], upper_bound-lower_bound, BODY_TYPE,  // the receive amount is based on the send amount of anothe rprocess
        //     comm);

        if (rank == 0){
            // receive 
        }

        world->old ^= 1;
    }

    /****************
    * NAIVE Post-processing
    *****************/

    // Gather results (memory inefficient for now)
    struct bodyType gathered_bodies[MAXBODIES]; // could malloc this... 
    struct world *gathered_world  = calloc(1, sizeof(*gathered_world));

    // naive first
    MPI_Gather(
        &world->bodies[lower_bound],  upper_bound-lower_bound, BODY_TYPE,
        gathered_bodies, upper_bound-lower_bound, BODY_TYPE,
        0, comm);

    gathered_world->bodyCt = world->bodyCt;
    for (int b = 0; b < gathered_world->bodyCt; b++) {
        gathered_world->bodies[b] = gathered_bodies[b];
    }

    if (rank == 0) {
        printf("RANK %d: Gathered\n", rank);
        print(gathered_world);
    }

    printf("RANK %d: Local\n", rank);
    print(world);

    // tear down mpi
    MPI_Finalize();
}

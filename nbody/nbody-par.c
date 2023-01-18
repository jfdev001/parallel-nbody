/*
Course     : Parallel Programming Practical
Course Code: X_400162
Professor  : Dr. H.E. Bal
Due        : 2023-02-03
Author     : Jared G. Frazier, 2795544
Description: Parallel n-body simulation

Usage:
```
## For nbody-sanity-check
# Base program:
prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi nbody/nbody-par 32 0 nbody.ppm 100000 

# Optimized program: 
prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi nbody/nbody-par 32 0 nbody.ppm 100000 --openmp

## For data collection only
prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi nbody/nbody-par 32 0 nbody.ppm 100000 --run-xps
prun -v -1 -np 2 -script $PRUN_ETC/prun-openmpi nbody/nbody-par 32 0 nbody.ppm 100000 --run-xps --openmp
```
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
#include <dirent.h>

#include <mpi.h>
#include <omp.h>

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

static void print(struct world *world);

/* MPI types and op
*/
MPI_Datatype BODY_TYPE;
MPI_Datatype WORLD_TYPE; 

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

/// @brief Modify array of receive counts using N bodies and P ranks.
void get_recvcounts(int *recvcounts, int N, int P) {
    // Get initial receive counts
    for (int rank = 0; rank < P; rank++) {
        recvcounts[rank] = N/P;
    }

    // Load balance
    for (int rank = N%P; rank > 0; rank--){
        recvcounts[rank] += 1;
    }
    
    return;
}

/// @brief Modify displacements to get offsets using receive counts.
void get_displacements(
    int *displacements, int *recvcounts, int P) {
    // First displacement is always 0
    displacements[0] = 0;

    for (int rank = 1; rank < P; rank++) {
        displacements[rank] = displacements[rank-1] + recvcounts[rank-1];
    }

    return;
}

/// @brief Modify lower and upper bound arrays to determine nbody indices
void get_bounds(
    int *lower_bounds, int *upper_bounds, 
    int *displacements, int *recvcounts,
    int P) {
    for (int rank = 0; rank < P; rank++) {
        lower_bounds[rank] = displacements[rank];
        upper_bounds[rank] = displacements[rank] + recvcounts[rank];
    }
    
    return;
}

/* N-body updates
*/

/// @brief Clear force accumulation variables for all bodies
static void
clear_forces(struct world *world, bool openmp)
{
    if (openmp) {
        #pragma omp parallel for
        for (int b = 0; b < world->bodyCt; ++b) {
            YF(world, b) = XF(world, b) = 0;
        }
    } else {
        for (int b = 0; b < world->bodyCt; ++b) {
            YF(world, b) = XF(world, b) = 0;
        }
    }
}

/// @brief  Compute forces on bodies within the given bounds
static void
compute_forces(
    struct world *world, int lower_bound, int upper_bound, 
    bool openmp)
{
    // Determine if openmp optimization is used
    if (openmp){
        // Uses negative force optimziation
        #pragma omp parallel for
        for (int b = lower_bound; b < upper_bound; ++b) {
            for (int c = b+1; c < upper_bound; ++c) {
                update_forces(world, b, c, true);
            }
        }       
        
        // Intra-world (using positions of bodies on other processes)
        // this cannot use the negative force optimization
        #pragma omp parallel for
        for (int b = lower_bound; b < upper_bound; ++b) {
            for (int c = 0; c < world->bodyCt; ++c){
                if (lower_bound <= c && c <= upper_bound) { continue; }
                update_forces(world, b, c, false);
            }
        }
    } else {
        // Inner-world (bodies on process) update
        // this uses the negative force optimization
        for (int b = lower_bound; b < upper_bound; ++b) {
            for (int c = b+1; c < upper_bound; ++c) {
                update_forces(world, b, c, true);
            }
        }        
        
        // Intra-world (using positions of bodies on other processes)
        // this cannot use the negative force optimization
        for (int b = lower_bound; b < upper_bound; ++b) {
            for (int c = 0; c < world->bodyCt; ++c){
                if (lower_bound <= c && c <= upper_bound) { continue; }
                update_forces(world, b, c, false);
            }
        }
    }
}

/// @brief Modify forces on bodies in place
void update_forces(struct world *world, int b, int c, bool intra_world) {
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

    // Update force for this worlds body
    XF(world, b) += xf;
    YF(world, b) += yf;

    // if c is in within the current world, then update it's forces
    if (intra_world) {
        XF(world, c) -= xf;
        YF(world, c) -= yf;
    }
}

/// @brief Compute velocities of bodies within given bounds
static void
compute_velocities(struct world *world, int lower_bound, int upper_bound, bool openmp)
{
    if (openmp) {
        #pragma omp parallel for
        for (int b = lower_bound; b < upper_bound; ++b) {
            update_velocities(world, b);
        }
    } else {
        for (int b = lower_bound; b < upper_bound; ++b) {
            update_velocities(world, b);
        }
    }
}

/// @brief Update velocities in place.
void update_velocities(struct world *world, int b) {
    double xv = XV(world, b);
    double yv = YV(world, b);
    double force = sqrt(xv*xv + yv*yv) * FRICTION;
    double angle = atan2(yv, xv);
    double xf = XF(world, b) - (force * cos(angle));
    double yf = YF(world, b) - (force * sin(angle));

    XV(world, b) += (xf / M(world, b)) * DELTA_T;
    YV(world, b) += (yf / M(world, b)) * DELTA_T;
}

/// @brief Compute positions of bodies within given bounds
static void
compute_positions(struct world *world, int lower_bound, int upper_bound, bool openmp)
{
    if (openmp) {
        #pragma omp parallel for
        for (int b = lower_bound; b < upper_bound; ++b) {
            update_positions(world, b);
        }
    } else {
        for (int b = lower_bound; b < upper_bound; ++b) {
            update_positions(world, b);
        }
    }
}

/// @brief Update positions in place
void update_positions(struct world *world, int b) {
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

/* Post-processing
*/

/// @brief Get the PRUN environment file name
/// @details 
/// [Get Files in Dir](https://www.tutorialspoint.com/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-cplusplus)
/// [C Regex](https://stackoverflow.com/questions/1085083/regular-expressions-in-c-examples)
char* get_prun_environment_fname(){
    bool match = false;
    char *fname;
    char msgbuf[100];
    
    // get lists of directories
    DIR *dr;
    struct dirent *en;
    dr = opendir(".");
    if (dr) {
        en = readdir(dr);
        while (en != NULL && !match){
            fname = en->d_name;
            if (strstr(fname, ".PRUN_ENVIRONMENT") != NULL){
                match = true;
            }
            en = readdir(dr);
        }
        closedir(dr);
    }
    return fname;
}

/// @brief Return an array s.t. `array = {n_nodes, n_cpus_per_node}`
/// @details 
/// Effective C, Ch. 8
/// [Cpp Reference: fgets](https://en.cppreference.com/w/c/io/fgets)
/// [Geeks for geeks: atoi](https://www.geeksforgeeks.org/c-program-for-char-to-int-conversion/)
/// [Stack Overflow: subsrings](https://stackoverflow.com/questions/12784766/check-substring-exists-in-a-string-in-c)
int* get_prun_compute_args(char *prun_env_fname, int *arg_array){
    int foundargs = 0;
    int argstrlen;
    char buffer[1000];
    int buffer_len;
    char compute_arg[256];
    int compute_arg_ix;

    // open prun file
    FILE *fptr = fopen(prun_env_fname, "r");

    // While the two compute args of interest have not been found
    // or until the whole file hsa been read
    while(foundargs < 2 && fgets(buffer, sizeof(buffer), fptr) != NULL) {

        // Check to see if the substrings of interest match...
        // then use the buffer length information to populate the
        // actual integer argument
        buffer_len = strlen(buffer);
        if (strstr(buffer, "PRUN_CPUS=") != NULL) {
            argstrlen = strlen("PRUN_CPUS=");
            compute_arg_ix = 0;
            for (int c = argstrlen; c < buffer_len; c++) {
                compute_arg[compute_arg_ix] = buffer[c];
                compute_arg_ix++;
            }
            arg_array[0] = atoi(compute_arg);
            foundargs++;

        } else if(strstr(buffer,"PRUN_CPUS_PER_NODE=") != NULL) {
            argstrlen = strlen("PRUN_CPUS_PER_NODE=");
            compute_arg_ix = 0;
            for (int c = argstrlen; c < buffer_len; c++) {
                compute_arg[compute_arg_ix] = buffer[c];
                compute_arg_ix++;
            }
            arg_array[1] = atoi(compute_arg);
            foundargs++;
        }
    }

    // PRUN multiplies the -np <arg> and -<> when computing PRUN_CPUS
    // therefore, dividing tells the actual arg passed to the script
    arg_array[0] /= arg_array[1]; 

    // close file
    fclose(fptr);
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

    // Build MPI types
    build_mpi_body_type();
    build_mpi_world_type();

    unsigned int lastup = 0;
    unsigned int secsup;
    int steps;                         // number of timesteps
    double start;
    double stop;
    double rtime;                      // run time of simulation
    double gflops;                     // floating pt ops
    struct filemap image_map;          // for graphics

    bool running_experiments = false;  // for logging results of experiments
    bool openmp = false;               // for using openmp or not
    int prun_compute_args[2];          // for -np <> and -<> args from PRUN_ENVIRONMENT

    int *displacements = NULL;
    int *recvcounts = NULL;
    int *lower_bounds = NULL;
    int *upper_bounds = NULL;
    int lower_bound;
    int upper_bound;

    // Allocate nbody world
    struct world *world = calloc(1, sizeof *world);
    if (world == NULL) {
        fprintf(stderr, "Cannot calloc(world)\n");
        exit(1);
    }

    /* Get Parameters 
    */

   // Indicate usage
    if (argc < 5) {
        if (rank == 0) {
            fprintf(
                stderr, 
                "Usage: %s num_bodies secs_per_update ppm_output_file steps %s\n",
                argv[0],
                "[--run-xps, --openmp]");
        }
        exit(1); 
    }

    // Set experiment flag
    if (argc > 5) {
        for (int i = 5; i < argc; i++){
            if (strcmp(argv[i], "--run-xps") == 0) {
                running_experiments = true;
            } else if (strcmp(argv[i], "--openmp") == 0) {
                openmp = true;
            }
        }
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

    // Print meta information to stderr
    if (rank == 0) {
        fprintf(stderr, "Running N-body with %i bodies and %i steps\n", world->bodyCt, steps);
    }

    // Initialize nbody world and then bcast it 
    if (rank == 0) {
        initialize_simulation_data(world);
    }

    MPI_Bcast(world, 1, WORLD_TYPE, 0, comm);

    // Determine bounds for computation
    recvcounts = malloc(size * sizeof(int));
    displacements = malloc(size * sizeof(int));
    lower_bounds = malloc(size * sizeof(int));
    upper_bounds = malloc(size * sizeof(int));

    get_recvcounts(recvcounts, world->bodyCt, size);
    get_displacements(displacements, recvcounts, size);
    get_bounds(lower_bounds, upper_bounds, displacements, recvcounts, size);

    lower_bound = lower_bounds[rank];
    upper_bound = upper_bounds[rank];

    /****************
    * Main processing
    *****************/
    MPI_Barrier(comm); // all procs reach here 

    // start timer
    if (rank == 0) {
        start = MPI_Wtime();
    }


    // Nbody algo
    for (int step = 0; step < steps; step++) {
        clear_forces(world, openmp);
        compute_forces(world, lower_bound, upper_bound, openmp);
        compute_velocities(world, lower_bound, upper_bound, openmp);
        compute_positions(world, lower_bound, upper_bound, openmp);

        // Gathers different bodies in partitions...
        // needed to get position information
        // NOTE: When attempting to do this with non-blocking communication
        // i get a segmentation fault... i think this is because the rest 
        // of the loop modifies the world->bodies member, which is not allowed?
        MPI_Allgatherv(
            MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
            world->bodies,
            recvcounts, displacements,
            BODY_TYPE, comm);

        world->old ^= 1;
    }

    // stop timer
    if (rank == 0) {
        stop = MPI_Wtime();
        rtime = stop - start;
    }

    /****************
    * Post-processing
    *****************/

    // NOTE: During the gathering of positions, I also gather the
    // the velocities (XV, YV) and forces (XF, YF)... thus any
    // world could be printed at the end and they will all have the
    // same data for those members

    // Print results
    if (rank == 0) {
        gflops = nr_flops(world->bodyCt, steps) / 1e9 / rtime;
        if (!running_experiments) {
            print(world);
            fprintf(stderr, "\nN-body took: %.3f seconds\n", rtime);
            fprintf(stderr, "Performance N-body: %.2f GFLOPS\n", gflops);
        } else {
            // For writing stdout to a file
            char *prun_env_fname = get_prun_environment_fname();
            get_prun_compute_args(prun_env_fname, prun_compute_args);

            // SIZE, NODES, CPUS_PER_NODE, NBODIES, RTIME, GLOPS
            // IMPORTANT: NODES AND CPUS_PER_NODE will be wrong if multiple experiment
            // scripts are running
            // OMP w/ 10k steps might sig kill runtime
            printf(
                "%d,%d,%d,%d,%.3f,%.2f\n", 
                size, prun_compute_args[0], prun_compute_args[1], world->bodyCt, rtime, gflops);
        }
    }

    // Cleanup
    MPI_Type_free(&BODY_TYPE);
    MPI_Type_free(&WORLD_TYPE);

    free(world);
    world = NULL;

    free(displacements);
    displacements = NULL;
    free(recvcounts);
    recvcounts = NULL;
    free(lower_bounds);
    lower_bounds = NULL;
    free(upper_bounds);
    upper_bounds = NULL;

    MPI_Finalize();
}

/*
Create three arrays:
(1) Forces computed as normal with the sequential version
(2) Forces computed as if this array was for P0
(3) Forces computed as if this array was for P1

Then combine the results of (2) and (3) into a single array
and see if they match the sequentially computed result.

NOTE: Will i need a function that approximates within some sort
of bound? Can i round to ease the potential for floating point.

$ # While in proto/
$ gcc -o test_2_compute_F test_2_compute_F.c -lm -g
$ ./test_2_compute_F     # wrong output
$ ./test_2_compute_F SUM # right output
*/


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <errno.h>

#define GRAVITY     1.1
#define FRICTION    0.01
#define MAXBODIES   10000
#define DELTA_T     (0.025/5000)
#define BOUNCE      -0.9
#define SEED        27102015


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

/// @brief The number of bodies will be determined by scattering at runtime
struct world {
    struct bodyType bodies[MAXBODIES];
    int                 bodyCt;
    int                 old;    // Flips between 0 and 1

    /*  Dimensions of space (very finite, ain't it?) */
    int                 xdim;
    int                 ydim;
};

void update_forces(struct world *world, int b, int c);
void compute_forces(struct world *world);
void initialize_data(struct world *world);
void print_xf(struct world *world, int b, int bodyCt);

int main(int argc, char **argv) {

    int bodyCt, xdim, ydim;
    bodyCt = 8;
    xdim = 10;
    ydim = 10;

    struct world *main_world = calloc(1, sizeof(*main_world));
    struct world *p0_world = calloc(1, sizeof(*p0_world));
    struct world *p1_world = calloc(1, sizeof(*p1_world));

    if (main_world == NULL || p0_world == NULL || p1_world == NULL) 
    { 
        printf("not enough mem\n"); exit(1); 
    }

    main_world->bodyCt = bodyCt;
    main_world->xdim = xdim;
    main_world->ydim = ydim;

    p0_world->bodyCt = bodyCt;
    p0_world->xdim = xdim;
    p0_world->ydim = ydim;

    p1_world->bodyCt = bodyCt;
    p1_world->xdim = xdim;
    p1_world->ydim = ydim;

    initialize_data(main_world);
    initialize_data(p0_world);
    initialize_data(p1_world);

    compute_forces(main_world);

    printf("Main World:\n");
    print_xf(main_world, 0, bodyCt);
    printf("\n");

    // Compute partition 0 forces
    for (int b = 0; b < bodyCt/2; b++) { // b in [0..4)
        for (int c = b+1; c < bodyCt; c++){ 
            update_forces(p0_world, b, c);
        }
    }

    // compute partition 1 forces
    for (int b = bodyCt/2; b < bodyCt; b++){ // b in [4, 8)
        for (int c = b+1; c < bodyCt; c++) { // misses previous C's... how to compensate?
            update_forces(p1_world, b, c);
        }
    }

    // add the latter half forces of the i-1 partition to the i partition
    if (argc > 1 && strcmp(argv[1], "SUM") == 0)
    {
        for (int b = bodyCt/2; b < bodyCt; b++){ // b in [4, 8)
            XF(p1_world, b) += XF(p0_world, b);
        }
    }

    printf("p0_world:\n");
    print_xf(p0_world, 0, bodyCt);
    printf("\n");
    
    printf("p1_world:\n");
    print_xf(p1_world, 0, bodyCt);
    printf("\n");

    return 0;
}

/// @brief Print the xforces of starting from desired begin/end points
void print_xf(struct world *world, int b, int bodyCt) 
{
    for (b; b < bodyCt; b++){
        printf("%.3lf ", XF(world, b));
    }
    printf("\n");
    return;
}

/// @brief Initialize the simulation data.
void initialize_data(struct world *world)
{
 /* Initialize simulation data */
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

/// @brief Compute the forces of bodies on one another.
void compute_forces(struct world *world)
{
    int b, c;

    /* Incrementally accumulate forces from each body pair,
       skipping force of body on itself (c == b)
    */
    for (b = 0; b < world->bodyCt; ++b) {
        for (c = b + 1; c < world->bodyCt; ++c) {
            update_forces(world, b, c);
        }
    }

    return;
}

/// @brief Update forces for bodies b and c for given world.
void update_forces(struct world *world, int b, int c) 
{
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

    return;
}
// mpicc -o redirect_stdout redirect_stdout.c; prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi ./redirect_stdout
#include <stdio.h>

int main() {
    fprintf(stderr, "fprintf to stderr\n");
    printf("To std out\n");
    return 0;
}
// mpicc -o open_f open_f.c; prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi ./open_f
#include <stdio.h>
#include <stdlib.h>

int main() {
    char fname[] = ".PRUN_ENVIRONMENT.23471.fs0";
    char buffer[1000];
    FILE *fptr = fopen(fname, "r");
    while (fgets(buffer, sizeof(buffer), fptr) != NULL){
        printf("%s\n", buffer);
    }
    return 0;
}
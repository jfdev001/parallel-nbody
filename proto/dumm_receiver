/*
Mallocing/callocing structs
gcc -o malloc_struct malloc_struct.c -g
./malloc_struct

References:
https://stackoverflow.com/questions/625138/structure-calloc-c
*/

#include <stdlib.h>
#include <stdio.h>

struct particle {
  double x;
  double y;
  double z;
};

struct simulation {
  struct particle particles[100];
};

int main(){
  struct simulation *sim_one = calloc(1, sizeof *sim_one);
  struct simulation *sim_two = malloc(sizeof(struct simulation));
  return 0;
}

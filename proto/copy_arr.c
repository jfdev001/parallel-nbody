/*
Play with array copying in C
gcc -o copy_arr copy_arr.c
prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi ./copy_arr
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void print_arr(int *arr, int size) {
    for (int i =0; i < size; i++){
        printf("%2d ", arr[i]);
    }
    printf("\n");
}

/// @brief arr1 is arr2
void myshallowcopy(int *arr1, int *arr2, int size){
    for (int i = 0; i < size; i++){
        int *ele2 = &arr2[i];
        arr1[i];
    }
}

/// @brief arr1 = arr2 
void mydeepcopy(int *arr1, int *arr2, int size) {
    memcpy(arr1, arr2, sizeof(arr2));
}

int main(int argc, char **argv) {

    // define
    int size = 5;
    int *arr1 = malloc(size * sizeof(int));
    int *arr2 = malloc(size * sizeof(int));

    // initialize
    srand(0);
    for (int i = 0; i < size; i ++) {
        arr1[i] = rand() % 5 + 0; // 0 -> 4
        arr2[i] = rand() % 5 + 5; // 5 -> 9
    }

    if (argc < 2) {
        fprintf(stderr, "usage: path/to/copy_arr [copy, deepcopy]\n");
        exit(1);
    }

    // inspect
    print_arr(arr1, size);
    print_arr(arr2, size);

    // perform copy
    if (strcmp(argv[1], "copy") == 0)  {
        myshallowcopy(arr1, arr2, size);

    } else if (strcmp(argv[1], "deepcopy") == 0) {
        mydeepcopy(arr1, arr2, size);
    }
    
    // inspect results
    printf("Result after copy\n");
    print_arr(arr1, size);
    print_arr(arr2, size);

    // Changing an element of arr 1
    printf("Changing an element of arr1\n");
    arr1[0] = -1;
    print_arr(arr1, size);
    print_arr(arr2, size);

}
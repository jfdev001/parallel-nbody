#!/usr/bin/bash
# @brief Run experiments using sequential script

OUTPUT=$1

if [ "$OUTPUT" == "" ]
then 
    echo "usage: ./bin/sequential-experiments.sh OUTPUT"
    exit 1
fi 

for replicate in 1 2 3
do 
    for N_BODIES in 512 1024 4096 10000
    do 
        prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi nbody/nbody-seq $N_BODIES 0 nbody.ppm 100 --run-xps >> $OUTPUT
    done 
done
#!/usr/bin/bash
# @brief Run experiments using sequential processing

OUTPUT=$1
ORIGINAL_SERIAL=$2  # 1 for original serial program, 0 for my parallel prog with -np 1

if [ "$OUTPUT" == "" ] || [ "$ORIGINAL_SERIAL" == "" ]
then 
    echo "usage: ./bin/sequential-experiments.sh <name of output file> [0,1]"
    exit 1
fi 

for replicate in 1 2 3
do 
    for N_BODIES in 512 1024 4096 10000
    do 
        if [ $ORIGINAL_SERIAL = 1 ]
        then 
            prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi nbody/nbody-seq $N_BODIES 0 nbody.ppm 100 --run-xps >> $OUTPUT
        else 
            prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi nbody/nbody-par $N_BODIES 0 nbody.ppm 100 --run-xps >> $OUTPUT
        fi 
    done 
done
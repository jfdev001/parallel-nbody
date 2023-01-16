#!/usr/bin/bash
# @brief Run experiments using parallel scripts

# Parse arguments
OPENMP=$1
OUTPUT=$2
N_ITERS=100

# Argument check
if [ "$OPENMP" == "" ] || [ "$OUTPUT" == "" ]
then 
    echo "usage: ./bin/experiments.sh [0, 1] <name of out file>"
    exit 1
fi 

# Determine if openmp will be used
if [ $OPENMP = 0 ]
then 
    OPENMP=""
else 
    OPENMP="--openmp"
fi

# Perform experiments with various n bodies
for replicate in 1 2 3
do
    for N_BODIES in 512 1024 4096 10000
    do 
        for NP in 2 4 6 8
        do 
            for CPU_PER_PROC in 1 4 16
            do 
                echo "N_BODIES=$N_BODIES, NP=$NP, CPU_PER_PROC=$CPU_PER_PROC, N_ITERS=$N_ITERS, OPENMP=$OPENMP"
                prun -v \
                    -$CPU_PER_PROC \
                    -np $NP \
                    -sge-script $PRUN_ETC/prun-openmpi nbody/nbody-par \
                    $N_BODIES 0 nbody.ppm $N_ITERS $OPENMP --run-xps >> $OUTPUT
            done
        done 
    done
done 
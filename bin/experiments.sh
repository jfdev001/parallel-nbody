#!/usr/bin/bash
# @brief Run experiments using parallel scripts

# Parse arguments
OPENMP=$1
OUTPUT=$2
MEASURE_COMM=$3
N_ITERS=100

# Argument check
if [ "$OPENMP" == "" ] || [ "$OUTPUT" == "" ] || [ "$MEASURE_COMM" == "" ]
then 
    echo "usage: ./bin/experiments.sh openmp={0, 1} <name of out file> measure_comm={0, 1}"
    exit 1
fi 

# Determine if openmp will be used
if [ $OPENMP = 0 ]
then 
    OPENMP=""
else 
    OPENMP="--openmp"
fi

if [ $MEASURE_COMM = 0 ]
then 
    MEASURE_COMM=""
else 
    MEASURE_COMM="--measure-comm"
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
                echo "N_BODIES=$N_BODIES, NP=$NP, CPU_PER_PROC=$CPU_PER_PROC, N_ITERS=$N_ITERS, OPENMP=$OPENMP, MEASURE_COMM=$MEASURE_COMM"
                prun -v \
                    -$CPU_PER_PROC \
                    -np $NP \
                    -sge-script $PRUN_ETC/prun-openmpi nbody/nbody-par \
                    $N_BODIES 0 nbody.ppm $N_ITERS $OPENMP $MEASURE_COMM --run-xps >> $OUTPUT
            done
        done 
    done
done 
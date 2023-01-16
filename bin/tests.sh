#!/usr/bin/bash
# @brief Performs thorough diff checks of parllel & serial output.
# @details The following minimal tests should be verified
# NP = {1, 2, 3, 4, 5, 6, 7, 8}
# CPU_PER_PROC = {1, 4, 16}
# (8, 16) -> NBODIES = Max(128, input)
# Need to ask about this because in principle, anything lower would
# mean that some processors do not have bodies

# get cli
NP=$1
CPU_PER_PROC=$2
N_BODIES=$3

# Name files based on args
REFERENCE_OUTPUT_FILE=tests/${N_BODIES}_MY_REF_OUTPUT
OUTPUT_FILE=tests/${NP}_${CPU_PER_PROC}_${N_BODIES}_mynbody.test.out
ERROR_FILE=tests/${NP}_${CPU_PER_PROC}_${N_BODIES}_mynbody.test.err
DIFF_FILE=tests/${NP}_${CPU_PER_PROC}_${N_BODIES}_mynbody.test.diff

echo "Running Tests: NP=$NP CPU_PER_PROC=$CPU_PER_PROC N_BODIES=$N_BODIES"

# Always compute sequential outputs
if ! test -f $REFERENCE_OUTPUT_FILE
then 
    prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi nbody/nbody-seq $N_BODIES 0 nbody.ppm 1000 | tee $REFERENCE_OUTPUT_FILE
fi

# Get outputs of parallel program
prun -v -$CPU_PER_PROC -np $NP -sge-script $PRUN_ETC/prun-openmpi nbody/nbody-par $N_BODIES 0 nbody.ppm 1000 2> $ERROR_FILE | tee $OUTPUT_FILE

# Get difference in outputs
diff $REFERENCE_OUTPUT_FILE $OUTPUT_FILE

# test differences
if test -s $DIFF_FILE;
then 
    echo "Status=FAILED" >> $OUTPUT_FILE
    echo "Status=FAILED" 
else 
    echo "STATUS=PASSED" >> $OUTPUT_FILE
    echo "STATUS=PASSED"
fi 


#!/usr/bin/bash

# @brief Comparisions of function outputs at different stages of development.
# of the sequential and parallel implementations
# @details Values are computed at every time step, however forces at 
# subsequent timesteps will be dependent on the positions and velocities updated
# in the previous timestep. Until the parallel implementation is complete,
# checking the output of the compute forces against the sequential implementation
# at a given timestep is impossible. Therefore, this script should simply check
# the computed force after a given timestep. Example, if 
# `./compute-forces-check 1`
# Then a single iteration of the loop should run and the forces should be checked
# i.e., 
# for t in timesteps:
#   clear_forces
#   compute_forces
#   ...
#   log(world.forces)
# Note that the forces are both x and y positions for a given body

# Define i/o files
REFERENCE_OUTPUT_FILE=bin/MY_REF_OUTPUT
OUTPUT_FILE=mynbody.test.out
# ERROR_FILE=mynbody.test.err
DIFF_FILE=mynbody.test.diff

# get cli
NP=$1
NUM_ITERS=$2
FLAG=$3

echo "Checking for correct output using np=$NP,NUM_ITERS=$NUM_ITERS, and FLAG=$FLAG"

# Get outputs of sequential program
echo "prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi nbody/nbody-seq-test 32 0 nbody.ppm $NUM_ITERS $FLAG"
prun -v -1 -np 1 -script $PRUN_ETC/prun-openmpi nbody/nbody-seq-test 32 0 nbody.ppm $NUM_ITERS $FLAG

# Get outputs of parallel program
echo "prun -v -1 -np $NP -sge-script $PRUN_ETC/prun-openmpi nbody/nbody-par 32 0 nbody.ppm $NUM_ITERS $FLAG | tee $OUTPUT_FILE"
prun -v -1 -np $NP -sge-script $PRUN_ETC/prun-openmpi nbody/nbody-par 32 0 nbody.ppm $NUM_ITERS $FLAG | tee $OUTPUT_FILE

# Get difference in outputs
diff $REFERENCE_OUTPUT_FILE $OUTPUT_FILE

# test differences
if test -s $DIFF_FILE;
then 
    echo "*** The program generated wrong output!" 
    echo "Diff between correct and found output:"
    cat $DIFF_FILE;
else 
    echo "output ok";
fi 
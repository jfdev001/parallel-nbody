#/usr/bin/bash
# @brief Wrapper for running more thorough tests

OPENMP=$1  # empty or openmp

# Vary parameters for test script
for NP in {1..8}; 
do
    for CPU_PER_PROC in 1 4 16; 
    do
        ./bin/tests.sh $NP $CPU_PER_PROC 128 $OPENMP;
    done;
done;

# Checks for failed tests
echo "Files that Failed the Tests:"
./bin/failed-tests.sh
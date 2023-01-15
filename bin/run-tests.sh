#/usr/bin/bash
# @brief Wrapper for running more thorough tests

# Vary parameters for test script
for NP in {1..8}; 
do
    for CPU_PER_PROC in 1, 4, 16; 
    do
        ./bin/tests.sh $NP $CPU_PER_PROC 128;
    done;
done;
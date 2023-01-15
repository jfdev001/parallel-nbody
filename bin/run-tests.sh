#/usr/bin/bash
# @brief Wrapper for running more thorough tests

# Vary parameters for other script
for NP in {1..8}; 
do
    for CPU_PER_PROC in 1, 4, 16; 
    do
        echo "($NP, $CPU_PER_PROC)";
    done;
done;
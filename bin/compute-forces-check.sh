#!/usr/bin/bash

# @brief Compares the compute forces of the sequential and parallel implementations
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
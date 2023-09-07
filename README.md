MPI + OpenMP assignment for the Parallel Programming Practical (PPP). 

Performs an n-celestial body simulation that is accelerated using both
MPI and optionally OpenMP. See the contents below for a description of
folders.

Content:

- nbody/ Contains sequential version of the N-body algorithm
  Create parallel version of the N-body algorithm there
  and name it nbody-par. 
- docs/ Report on findings for speedups and efficiency.
- bin/ Contains nbody-sanity-check (comparison with expected output) as well
  as testing scripts.

References

- [Youtube: MPI Derived Types](https://www.youtube.com/watch?v=x_GZtMCr4W4)
- [How to Debug using GDB](https://u.osu.edu/cstutorials/2018/09/28/how-to-debug-c-program-using-gdb-in-6-simple-steps/)
- [Examples using MPI_GATHER, MPI_GATHERV](https://www.mpi-forum.org/docs/mpi-1.1/mpi-11-html/node70.html)

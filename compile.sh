module load apps/lammps/gpu

module load compiler/mpi/mpich/3.2/gnu
nvcc gpu_code.cu -lmpi


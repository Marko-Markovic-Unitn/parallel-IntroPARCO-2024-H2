# parallel-IntroPARCO-2024-H2

## Reproducibility Instructions

This repository contains all the necessary files and instructions to reproduce the results presented in the report. For the OpenMP results, the corresponding code can be found in the `parallel-IntroPARCO-2024-H1` repository. Follow the instructions below to compile and run the programs.

---

## Compilation Instructions

To compile the code, follow these steps:

1. **Ensure Necessary Compilers and Dependencies Are Installed:**
   - **GCC/G++** (with OpenMP support) for shared-memory parallelization.
   - **mpicxx** for MPI-based distributed memory parallelization.

2. **Compilation Commands:**
   - **Parallel Implementation (OpenMP):**
     ```bash
     g++ -O2 -fopenmp -march=native -ftree-vectorize -fprefetch-loop-arrays -fopt-info-vec-optimized -o matrix_parallel deliverable.c
     ```
     If running locally, the number of threads can be set as follows:
     ```bash
     OMP_NUM_THREADS=N ./matrix_parallel [matrix_size]
     ```
     Alternatively, you can set the number of threads directly in the C code:
     ```c
     omp_set_num_threads(4); // Example: Set the number of threads to 4
     ```
     If running on a cluster, ensure the following modules are loaded:
     - `gcc` (e.g., gcc91 as used in the report)
     - `mpich` (e.g., mpich-3.2.1--gcc-9.1.0 as used in the report)

     Create a `.pbs` file to configure the job. For example:
     ```bash
     #!/bin/bash
     #PBS -N matrix_parallel
     #PBS -o matrix_parallel.out
     #PBS -e matrix_parallel.err
     #PBS -q short_cpuQ
     #PBS -l walltime=0:02:00
     #PBS -l select=1:ncpus=1:mpiprocs=1:mem=1gb

     # Set the number of threads for OpenMP
     export OMP_NUM_THREADS=1

     # Change to the working directory (if needed)
     # cd /home/user/EsameHPC

     # Run the OpenMP program
     mpiexec -np 1 ./matrix_parallel 1024
     ```

   - **Distributed Implementation (MPI):**
     ```bash
     mpicxx -O2 -march=native -ftree-vectorize -fprefetch-loop-arrays -fopt-info-vec-optimized -o matrix_parallel deliverable.c
     ```

---

## Execution Instructions

Once the programs are compiled, execute them as follows:

1. **Parallel Implementation (OpenMP):**
   - If the number of threads was set in the code file:
     ```bash
     ./matrix_parallel [matrix_size]
     ```
   - Otherwise, specify the number of threads during execution:
     ```bash
     OMP_NUM_THREADS=N ./matrix_parallel [matrix_size]
     ```
   - If you use pbs instead:
     ```bash
	 qsub pbs_file.pbs
	 ```


2. **Distributed Implementation (MPI):**
   - Run using the following command if manually):
     ```bash
     mpirun -np [number_CPUs] ./matrix_transpose_parallel [matrix_size]
     ```
   - If you use pbs instead:
     ```bash
	 qsub pbs_file.pbs
	 ```

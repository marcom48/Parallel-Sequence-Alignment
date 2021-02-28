# HPC Parallel Sequence Alignment
This repository contains two programs that perform parallel variations of solving the [Sequence Alignment problem](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/sequence-alignment). These programs were part of a two part university project for a parallel & multicore computing class, where students were given access to a dozen nodes in the University of Melbourne's HPC cluster, [Spartan](https://dashboard.hpc.unimelb.edu.au/), and were graded based on their relative execution time in the cohort. Both of these files ranked first for fastest time.

## Generate Test Data
### Sequence Alignment
```bash
python seq_gen.py
```
### k-Sequence Alignment
```bash
python k_seq_gen.py
```

## Sequence Alignment (OpenMP)
```bash
g++ -fopenmp -o mmarasco-seqalignomp -O3 mmarasco-seqalignomp.cpp
./mmarasco-seqalignomp < 30000.dat
```

## k-Sequence Alignment (OpenMP & OpenMPI)
```bash
mpicxx -o mmarasco-seqalkway mmarasco-seqalkway.cpp -fopenmp -O3
mpiexec mmarasco-seqalkway < 12-30000.dat
```

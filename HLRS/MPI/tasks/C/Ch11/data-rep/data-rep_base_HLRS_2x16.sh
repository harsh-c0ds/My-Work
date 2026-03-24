#!/bin/bash

#PBS -N datarep_base
#PBS -l select=2:node_type=rome:mpiprocs=16
#PBS -l walltime=00:02:00
 
# commands for HAWK cluster

cd $PBS_O_WORKDIR

# Compile @login node:   mpicc  -o data-rep_base data-rep_base.c
#                        mpif08 -o data-rep_base data-rep_base_30.f90

mpirun -n 32 ./data-rep_base

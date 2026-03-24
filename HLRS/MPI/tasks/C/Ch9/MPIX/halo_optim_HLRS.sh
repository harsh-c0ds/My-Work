#!/bin/bash
#PBS -l nodes=8:ppn=24
#PBS -l walltime=00:01:00             
#PBS -j oe
  
# Change to the direcotry that the job was submitted from
cd $PBS_O_WORKDIR

################################################################################
#
# Compile on mom-node:   cc -o my-program.exe my-program.c
#        in this case:   cc -o halo_optim.exe halo_optim.c MPIX_*.c
#                  or:   cc -o halo_optim.exe halo_irecv_send_toggle_3dim_grid_solution.c MPIX_*.c
#                        fort -o my-program.exe my-program.f90
#     
# Submit to queuing system:   qsub job_mpi.sh
#             in this case:   qsub halo_optim_HLRS.sh
#                   check:    qstat JOB_ID
#                   cancel:   qdel  JOB_ID
#
#           find output in:   halo_optim_HLRS.sh.oJOB_ID
#
################################################################################

date
 time aprun -n 192 -N 24 ./halo_optim.exe < input-optim.txt
date

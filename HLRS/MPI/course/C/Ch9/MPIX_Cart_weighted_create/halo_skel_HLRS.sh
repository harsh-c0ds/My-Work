#!/bin/bash
#PBS -l nodes=8:ppn=24
#PBS -l walltime=00:01:00             
#PBS -j oe
  
# Change to the direcotry that the job was submitted from
cd $PBS_O_WORKDIR

################################################################################
#
# Compile on mom-node:   cc -o my-program.exe my-program.c
#        in this case:   cc -o halo_skel.exe halo_irecv_send_toggle_3dim_grid_skel.c MPIX_*.c
#                        fort -o my-program.exe my-program.f90
#     
# Submit to queuing system:   qsub job_mpi.sh
#             in this case:   qsub halo_skel_HLRS.sh
#                   check:    qstat JOB_ID
#                   cancel:   qdel  JOB_ID
#
#           find output in:   halo_skel_HLRS.sh.oJOB_ID
#
################################################################################

date
 time aprun -n 192 -N 24 ./halo_skel.exe < input-skel.txt 
date

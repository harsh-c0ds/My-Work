#!/bin/bash
#SBATCH -J job                          # SLURM_JOB_NAME      ==> jobname
#SBATCH -N 12                           # SLURM_JOB_NUM_NODES ==> @course [1-4]
#SBATCH --tasks-per-node=16             # SLURM_NTASKS_PER_NODE [default: 32/32]
#SBATCH --clusters=ivymuc               # ivymuc
###SBATCH --reservation=hhyp1w18_course   # node-reservation during course only
#SBATCH --export=NONE                   # do not inherit the submission env
#SBATCH --get-user-env                  # ivymuc: recommended run system profile
#SBATCH --time=00:01:00                 # time limit         ==> @course [1 min]

################################################################################
#
# Compile @login node:   mpiicc -o my-program.exe my-program.c
#        in this case:   mpiicc -o halo_optim.exe halo_optim.c MPIX_*.c
#                  or:   mpiicc -o halo_optim.exe halo_irecv_send_toggle_3dim_grid_solution.c MPIX_*.c
#                        mpiifort -o my-program.exe my-program.f90
#     
# Submit to queuing system:   sbatch job_mpi.sh
#             in this case:   sbatch halo_optim_LRZ.sh
#                   check:    squeue --clusters=ivymuc
#                   cancel:   scancel --clusters=ivymuc JOB_ID
#
#           find output in:   slurm-JOB_ID.out
#
################################################################################

# Set (max) number of MPI processes
  export MPI_PROCESSES=$SLURM_NTASKS    # ivymuc = needed@h.: --tasks-per-node=#

# Print out debugging information (Intel MPI) / verbose output (Intel OpenMP)
  export VERBOSE=noverbose          # noverbose output (use within KMP_AFFINITY)
# export VERBOSE=verbose            # verbose output (use within KMP_AFFINITY)
# export KMP_AFFINITY=$VERBOSE      # list topology map, maybe need some more...
# export I_MPI_DEBUG=4              # debugging info (MPI, 4 = pinning info)

# Default settings on ivymuc:
# export OMP_NUM_THREADS=1                                    # ivymuc = default
# export KMP_AFFINITY=$VERBOSE,granularity=thread,compact,1,0 # ivymuc = default
# export I_MPI_PIN_DOMAIN=auto                                # ivymuc = default

################################################################################
# Take care of correct placement of MPI processes on ivymuc
################################################################################

# pure MPI: mpirun --> ORDERED (FILL SOCKETS FIRST):
# export I_MPI_PIN_PROCESSOR_LIST=...                # does not work on ivymuc !
# export I_MPI_PIN_PROCESSOR_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  export I_MPI_PIN_DOMAIN=[1,2,4,8,10,20,40,80,100,200,400,800,1000,2000,4000,8000]

date
  mpirun -n $MPI_PROCESSES ./halo_optim.exe < input-optim.txt 
date


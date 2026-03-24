#!/bin/bash
#SBATCH -J JOBNAME                  # SLURM_JOB_NAME         ==> jobname
#SBATCH -N 12                       # SLURM_JOB_NUM_NODES    ==> @course [1-4]
#SBATCH --tasks-per-node=16         # SLURM_NTASKS_PER_NODE     [default: 16/32]
#SBATCH --export=NONE               # do not inherit the submission environment
###SBATCH --get-user-env              # run system profile             [?maybe?]
#SBATCH --time=00:01:00             # time limit             ==> @course [1 min]

module purge                        # always start with a clean environment
module load intel/18 intel-mpi/2018 # load modules needed (? mpi compiler names)

################################################################################
#
# Load modules @login node:   module load intel/18 intel-mpi/2018
#
# Compile @login node:        
# mpiicc -o halo_skel.exe halo_irecv_send_toggle_3dim_grid_skel.c MPIX_*c
#     
# Submit to queuing system:   sbatch JOBFILE
#                   check:    squeue -u $USER | grep JOBNAME
#                   cancel:   scancel JOB_ID
#
#           find output in:   slurm-JOB_ID.out
#
################################################################################

################################################################################
# Take care of correct placement of MPI processes on vsc3 (mem_0064 partition)
################################################################################

# Print out debugging information (Intel MPI)
# export I_MPI_DEBUG=4              # debugging info (MPI, 4 = pinning info)

# Set (max) number of MPI processes
  export MPI_PROCESSES=$SLURM_NTASKS

# pure MPI: mpirun --> ORDERED (FILL SOCKETS FIRST):
  export I_MPI_PIN_PROCESSOR_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 #default

echo $SLURM_JOB_NODELIST

date
  mpirun -n $MPI_PROCESSES ./halo_skel.exe < input-skel.txt
date


#!/usr/bin/env python3

################################################################# 
#                                                               # 
#  This file has been written as a sample solution to an        # 
#  exercise in a course given at the High Performance           # 
#  Computing Centre Stuttgart (HLRS).                           # 
#  The examples are based on the examples in the MPI course of  # 
#  the Edinburgh Parallel Computing Centre (EPCC).              # 
#  It is made freely available with the understanding that      # 
#  every copy of this file must include this header and that    # 
#  HLRS and EPCC take no responsibility for the use of the      # 
#  enclosed teaching material.                                  # 
#                                                               # 
#  Authors: Joel Malard, Alan Simpson,            (EPCC)        # 
#           Rolf Rabenseifner, Traugott Streicher,              #
#           Tobias Haas (HLRS)                                  #
#                                                               # 
#  Contact: rabenseifner@hlrs.de                                # 
#                                                               # 
#  Purpose: A program to try out one-sided communication        # 
#           with window=rcv_buf and MPI_PUT to put              # 
#           local snd_buf value into remote window (rcv_buf).   # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

# ATTENTION!
# mpi4py initializes MPI with 'MPI_THREAD_MULTIPLE'. This might not work (e.g. 
# message 'MPI_ERR_WIN: invalid window') for your installation on, for instance,
# older OpenMPI versions. You can  try to initialize it in single
# thread mode by uncommenting the following two lines
# import mpi4py
# mpi4py.rc.thread_level = 'single' # or perhaps 'serialized'

from mpi4py import MPI
import numpy as np

np_dtype = np.intc
rcv_buf = np.empty((),dtype=np_dtype)
status = MPI.Status()

comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()
my_rank = comm_world.Get_rank()

right = (my_rank+1)      % size;
left  = (my_rank-1+size) % size;

# Create the window.
win = MPI.Win.Create(_____________________________________________________________________________)

sum = 0;
snd_buf = np.array(my_rank, dtype=np_dtype)

for i in range(size):
   request = comm_world.Issend((snd_buf, 1, MPI.INT), right, 17)
   comm_world.Recv            ((rcv_buf, 1, MPI.INT), left,  17, status)
   request.Wait(status)
   
   np.copyto(snd_buf, rcv_buf)
   sum += rcv_buf

print(f"PE{my_rank}:\tSum = {sum}")


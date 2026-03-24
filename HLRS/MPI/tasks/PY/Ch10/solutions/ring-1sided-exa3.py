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
#           With Start-Post-Complete-Wait synchronization.      #
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

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()

size = comm_world.Get_size()

right = (my_rank+1)      % size
left  = (my_rank-1+size) % size
# ... this SPMD-style neighbor computation with modulo has the same meaning as:
# right = my_rank + 1
# if (right == size):
#    right = 0
# left = my_rank - 1  
# if (left == -1):
#    left = size-1

# Create the window.

win = MPI.Win.Create(memory=rcv_buf, disp_unit=rcv_buf.itemsize, info=MPI.INFO_NULL, comm=comm_world)

# Get nearest neighbour rank as group-arguments.
grp_world = comm_world.Get_group()
grp_left  = grp_world.Incl([left])
grp_right = grp_world.Incl([right])
grp_world.Free()

sum = 0
snd_buf = np.array(my_rank, dtype=np_dtype)

for i in range(size):
   win.Post(grp_left, MPI.MODE_NOSTORE)
   win.Start(grp_right, 0)
   win.Put((snd_buf, 1, MPI.INT), right, (0, 1, MPI.INT))
   win.Complete()
   win.Wait()

   np.copyto(snd_buf, rcv_buf)
   sum += rcv_buf

print(f"PE{my_rank}:\tSum = {sum}")

win.Free()

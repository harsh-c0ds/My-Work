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
# # This code will work with mpi4py 3.0.0 and above, see comment below.

from mpi4py import MPI
import numpy as np

np_dtype = np.intc

comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()
my_rank = comm_world.Get_rank()

right = (my_rank+1)      % size
left  = (my_rank-1+size) % size

# Allocate the window.
win = MPI.Win.Allocate(np_dtype(0).itemsize, np_dtype(0).itemsize, MPI.INFO_NULL, comm_world)
# The buffer interface is not implemented for the Win classe prior to version 3.0.0. 
# This code will work with mpi4py 3.0.0 and above.
rcv_buf = np.frombuffer(win, dtype=np_dtype)
rcv_buf = np.reshape(rcv_buf,())
 
sum = 0;
snd_buf = np.array(my_rank, dtype=np_dtype)

for i in range(size):
   win.Fence(MPI.MODE_NOSTORE | MPI.MODE_NOPRECEDE)
   win.Put((snd_buf, 1, MPI.INT), right, (0, 1, MPI.INT))
   win.Fence(MPI.MODE_NOSTORE | MPI.MODE_NOPUT | MPI.MODE_NOSUCCEED)

   np.copyto(snd_buf, rcv_buf)
   sum += rcv_buf

print(f"PE{my_rank}:\tSum = {sum}")

win.Free()

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
# This code will work with mpi4py 3.0.0 and above, see comment below.

from mpi4py import MPI
import numpy as np

np_dtype = np.intc
# Just dummy values.
my_rank_sm = -1000
size_sm = -1000

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

# if (my_rank_world == 0): # PLEASE ACTIVATE IN EXERCISE 1
#    if (size_sm == size_world):
#       print("MPI_COMM_WORLD consists of only one shared memory region")
#    else:
#       print("MPI_COMM_WORLD is split into 2 or more shared memory islands")

right = (my_rank_world+1)         % size_world
left  = (my_rank_world-1+size_world) % size_world

# Allocate the window.
win = MPI.Win.Allocate(np_dtype(0).itemsize*1, np_dtype(0).itemsize, MPI.INFO_NULL, comm_world)
# The buffer interface is not implemented for the Win classe prior to version 3.0.0. 
# This code will work with mpi4py 3.0.0 and above.
rcv_buf = np.frombuffer(win, dtype=np_dtype)
rcv_buf = np.reshape(rcv_buf,())

sum = 0
snd_buf = np.array(my_rank_world, dtype=np_dtype)

for i in range(size_world):
   win.Fence(MPI.MODE_NOSTORE | MPI.MODE_NOPRECEDE)
   win.Put((snd_buf, 1, MPI.INT), right, (0, 1, MPI.INT))
   win.Fence(MPI.MODE_NOSTORE | MPI.MODE_NOPUT | MPI.MODE_NOSUCCEED)

   np.copyto(snd_buf,rcv_buf)
   sum += rcv_buf

print("World: {} of {} \tcomm_sm: {} of {} \tSum = {}".format(
      my_rank_world, size_world, my_rank_sm, size_sm, sum))

win.Free()

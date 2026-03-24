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

status = MPI.Status()
snd_dummy = np.zeros((),dtype=np_dtype)
rcv_dummy = np.zeros_like(snd_dummy)

comm_world = MPI.COMM_WORLD
my_rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

comm_sm = comm_world.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
my_rank_sm = comm_sm.Get_rank()
size_sm = comm_sm.Get_size()
if (my_rank_world == 0):
   if (size_sm == size_world):
      print("MPI_COMM_WORLD consists of only one shared memory region")
   else:
      print("MPI_COMM_WORLD is split into 2 or more shared memory islands")

right = (my_rank_sm+1)         % size_sm
left  = (my_rank_sm-1+size_sm) % size_sm

# Allocate the window.
win = MPI.Win.Allocate_shared(np_dtype(0).itemsize*1, np_dtype(0).itemsize, MPI.INFO_NULL, comm_sm)
# The buffer interface is not implemented for the Win classe prior to version 3.0.0. 
# This code will work with mpi4py 3.0.0 and above.
# We define an memory object with the rank 0 process' base address and length up to the last 
# element of the shared memory allocated by Allocate_shared. 
(buf_zero, itemsize) = win.Shared_query(0)
assert itemsize == MPI.INT.Get_size()
assert itemsize == np_dtype(0).itemsize
buf = MPI.memory.fromaddress(buf_zero.address, size_sm*1*itemsize)
# We use this memory object and consider it as an numpy ndarray
rcv_buf = np.frombuffer(buf, dtype=np_dtype)

win.Lock_all(MPI.MODE_NOCHECK)

sum = 0
snd_buf = np.array(my_rank_sm, dtype=np_dtype)

for i in range(size_sm):
   #  ... The local Win_syncs are needed to sync the processor and real memory.
   #  ... The following pair of syncs is needed that the read-write-rule is fulfilled.
   win.Sync()

   #    ... tag=17: posting to left that rcv_buf is exposed to left, i.e.,
   #            the left process is now allowed to store data into the local rcv_buf
   rq = comm_sm.Irecv((rcv_dummy, 0, MPI.INT), right, 17)
   comm_sm.Send((snd_dummy, 0, MPI.INT), left,  17)
   rq.Wait(status)

   win.Sync()


   # MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win);
   #   ... is substituted by (store into right neighbor's rcv_buf):
   rcv_buf[right] = snd_buf


   #  ... The following pair of syncs is needed that the write-read-rule is fulfilled.
   win.Sync()

   #  ... The following communication synchronizes the processors in the way
   #      that the origin processor has finished the store
   #      before the target processor starts to load the data.
   #  ... tag=18: posting to right that rcv_buf was stored from left
   rq = comm_sm.Irecv((rcv_dummy, 0, MPI.INT), left,  18)
   comm_sm.Send((snd_dummy, 0, MPI.INT), right, 18)
   rq.Wait(status)

   win.Sync()

  
   snd_buf = rcv_buf[my_rank_sm]
   sum += rcv_buf[my_rank_sm]

print("World: {} of {} \tcomm_sm: {} of {} \tSum = {}".format(
       my_rank_world, size_world, my_rank_sm, size_sm, sum))

win.Unlock_all()
win.Free()

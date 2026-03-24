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

from mpi4py import MPI
import numpy as np

np_dtype = np.intc

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
(buf_zero, itemsize) = win.Shared_query(0)
assert itemsize == MPI.INT.Get_size()
assert itemsize == np_dtype(0).itemsize
buf = MPI.memory.fromaddress(buf_zero.address, size_sm*1*itemsize)
rcv_buf = np.frombuffer(buf, dtype=np_dtype)

win.Lock_all(MPI.MODE_NOCHECK)

win_signal_A = MPI.Win.Allocate_shared(np_dtype(0).itemsize*1, np_dtype(0).itemsize, MPI.INFO_NULL, comm_sm)
(buf_zero, itemsize) = win_signal_A.Shared_query(0)
assert itemsize == MPI.INT.Get_size()
assert itemsize == np_dtype(0).itemsize
buf = MPI.memory.fromaddress(buf_zero.address, size_sm*1*itemsize)
signal_A_buf = np.frombuffer(buf, dtype=np_dtype)
win_signal_A.Fence(0) # 0 is default, could be omitted.
win_signal_A.Lock_all(MPI.MODE_NOCHECK)

win_signal_B = MPI.Win.Allocate_shared(np_dtype(0).itemsize*1, np_dtype(0).itemsize, MPI.INFO_NULL, comm_sm)
(buf_zero, itemsize) = win_signal_B.Shared_query(0)
assert itemsize == MPI.INT.Get_size()
assert itemsize == np_dtype(0).itemsize
buf = MPI.memory.fromaddress(buf_zero.address, size_sm*1*itemsize)
signal_B_buf = np.frombuffer(buf, dtype=np_dtype)
win_signal_B.Fence(0)
win_signal_B.Lock_all(MPI.MODE_NOCHECK)

win.Sync() # called by the origin process
comm_world.Barrier()
win.Sync() # called by the target process

sum = 0
snd_buf = np.array(my_rank_sm, dtype=np_dtype)

for i in range(size_sm):
   #  ... The local Win_syncs are needed to sync the processor and real memory.
   #  ... The following pair of syncs is needed that the read-write-rule is fulfilled.
   #  rcv_buf is now exposed, i.e., the reading of previous value is finished
   win.Sync() # called by the target process

   # ... posting to left that rcv_buf is exposed to left, i.e.,
   # the left process is now allowed to store data into the local rcv_buf
   signal_A_buf[left] = 1
   while(signal_A_buf[my_rank_sm]==0):
      pass # IDLE
   signal_A_buf[my_rank_sm] = 0
   #  ... the two syncs for the both signals  can be removed, because
   #  the following  MPI_Win_sync(win) will sync all three variables 
   #  MPI_Win_sync(win_signal_A)
   #  MPI_Win_sync(win_signal_B)

   win.Sync() # called by the origin process
   #  rcv_buf is now writable by the origin process


   # MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win)
   #   ... is substituted by (with offset "right-my_rank" to store into right neighbor's rcv_buf):
   rcv_buf[right] = snd_buf # the origin process writes into the target rcv_buf

   #  ... The following pair of syncs is needed that the write-read-rule is fulfilled.
   #  writing of rcv_buf from remote is finished
   win.Sync() # called by the origin process

   # ... The following communication synchronizes the processors in the way
   #     that the origin processor has finished the store
   #     before the target processor starts to load the data.
   # ... posting to right that rcv_buf was stored from left
   signal_B_buf[right] = 1
   while (signal_B_buf[my_rank_sm]==0):
      pass # IDLE
   signal_B_buf[my_rank_sm] = 0
   #  ... the two syncs for the both signals  can be removed, because
   #     the following  MPI_Win_sync(win) will sync all three variables
   #  MPI_Win_sync(win_signal_B); 
   #  MPI_Win_sync(win_signal_A); 

   win.Sync() # called by the target process
   #  rcv_buf is now locally (i.e., by the origin process) readable
  
   snd_buf = rcv_buf[my_rank_sm]
   sum += rcv_buf[my_rank_sm]

print("World: {} of {} \tcomm_sm: {} of {} \tSum = {}".format( 
       my_rank_world, size_world, my_rank_sm, size_sm, sum))

win.Unlock_all()
win_signal_A.Unlock_all()
win_signal_B.Unlock_all()

win.Free()

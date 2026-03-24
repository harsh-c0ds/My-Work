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
#  Purpose: The original pt-to-pt halo communication in a ring  #
#           through all processes should be kept between the    #
#           sub-islands and substituted with shared memory store#
#           within the sub-islands.                             #
#           Take care that the synchronization does not deadlock#
#           even if the sub-islands contain only one process.   #
#           Instead of the comm_sm shared memory islands, we    #
#           use smaller sub-islands, because running on a       #
#           shared system, one can still have more then one     #
#           such sub-islands. In this exercise, we therefore    #
#           communicate through pt-to-pt within MPI_COMM_WORLD  #
#           or through shared memory assignments in comm_sm_sub.#
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

comm_world = MPI.COMM_WORLD
my_rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

# original calculation of the neighbors within MPI_COMM_WORLD
right = (my_rank_world+1)            % size_world
left  = (my_rank_world-1+size_world) % size_world

comm_sm = comm_world.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
my_rank_sm = comm_sm.Get_rank()
size_sm = comm_sm.Get_size()
if (my_rank_world == 0):
   if (size_sm == size_world):
      print("MPI_COMM_WORLD consists of only one shared memory region")
   else:
      print("MPI_COMM_WORLD is split into 2 or more shared memory islands")

# Splitting comm_sm into smaller sub-islands. Of course, they are also shared memory islands.
size_sm_sub = size_sm // 2  # One may spilt also into more than 2 sub-islands
if (size_sm_sub == 0):
   size_sm_sub = 1
color = my_rank_sm // size_sm_sub
comm_sm_sub = comm_sm.Split(color, 0)
my_rank_sm_sub = comm_sm_sub.Get_rank()
size_sm_sub = comm_sm_sub.Get_size()

# Allocate the window within the sub-islands.
win = MPI.Win.Allocate_shared(np_dtype(0).itemsize*1, np_dtype(0).itemsize, MPI.INFO_NULL, comm_sm_sub)
# The buffer interface is not implemented for the Win classe prior to version 3.0.0. 
# This code will work with mpi4py 3.0.0 and above.
# We define an memory object with the root (0) process' base address and length up to the last 
# element of the shared memory allocated by Allocate_shared.
(buf_zero, itemsize) = win.Shared_query(0)
assert itemsize == MPI.INT.Get_size()
assert itemsize == np_dtype(0).itemsize 
buf = MPI.memory.fromaddress(buf_zero.address, itemsize*1*size_sm_sub)
# We use this memory object and consider it as an numpy ndarray
rcv_buf = np.frombuffer(buf, dtype=np_dtype)

# Is my neighbor in MPI_COMM_WORLD accessible within comm_sm_sub?
grp_world = comm_world.Get_group()
grp_sm_sub = comm_sm_sub.Get_group()

# check for left neighbor: (for simplification, two calls are used instead of setting up an array of ranks)
left_sm_sub = MPI.Group.Translate_ranks(grp_world, [left], grp_sm_sub)
# if left_sm_sub != MPI.UNDEFINED then receive from left is possible through comm_sm_sub

# check for right neighbor:
right_sm_sub = MPI.Group.Translate_ranks(grp_world, [right], grp_sm_sub)
# if right_sm_sub != MPI.UNDEFINED then send to right is possible through comm_sm_sub

sum = 0
snd_buf = np.array(my_rank_world, dtype=np_dtype)

for i in range(size_world):
   if(right_sm_sub[0] == MPI.UNDEFINED):
      request = comm_world.Issend((snd_buf, 1, MPI.INT), right, 17)
   if(left_sm_sub[0] == MPI.UNDEFINED):
      comm_world.Recv((rcv_buf[my_rank_sm_sub:my_rank_sm_sub+1], 1, MPI.INT), left,  17, status)
   if(right_sm_sub[0] == MPI.UNDEFINED):
      request.Wait(status)
  
   # collective, therefore all processes in comm_sm_sub must call
   win.Fence(0)  # workaround: no assertions
   # MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win);
   #   ... is substituted by (store into right neighbor's rcv_buf):
   if(right_sm_sub[0] != MPI.UNDEFINED):
      rcv_buf[my_rank_sm_sub+1] = snd_buf
   # collective, therefore all processes in comm_sm_sub must call
   win.Fence(0)  # workaround: no assertions

   np.copyto(snd_buf, rcv_buf[my_rank_sm_sub])
   sum += rcv_buf[my_rank_sm_sub]

print("World: {} of {} l/r={}/{} comm_sm: {} of {} comm_sm_sub: {} of {} l/r={}/{} Sum = {}".format( 
       my_rank_world,size_world, left,right, my_rank_sm,size_sm, 
       my_rank_sm_sub,size_sm_sub, left_sm_sub[0],right_sm_sub[0],  sum))

win.Free()

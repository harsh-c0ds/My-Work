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
#  Purpose: A program to try MPI_Irecv and MPI_Issend.          # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

to_right = 201
rcv_buf = np.empty((), dtype=np.intc)

arr_status = [ MPI.Status() ]*2
arr_request = [None, None]      # Not strictly necessary, since we will assigne other objects later. Could also simply append to empty list.

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

right = (my_rank+1)      % size;
left  = (my_rank-1+size) % size;
# ... this SPMD-style neighbor computation with modulo has the same meaning as:
# right = my_rank + 1          
# if (right == size):
#    right = 0
#    left = my_rank - 1
# if (left == -1):
#    left = size-1

sum = 0
snd_buf = np.array(my_rank, dtype=np.intc)

for i in range(size):
   arr_request[0] = comm_world.Irecv((rcv_buf, 1, MPI.INT), 
                                     source=left,  tag=to_right)
                              
   arr_request[1] = comm_world.Issend((snd_buf, 1, MPI.INT), 
                                      dest=right, tag=to_right)
   
   MPI.Request().Waitall(arr_request,arr_status)
    
   np.copyto(snd_buf, rcv_buf)
   sum += snd_buf
    
print(f"PE{my_rank}:\tSum = {sum}")

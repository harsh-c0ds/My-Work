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
#  Purpose: A WRONG ring communication program                  # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

rcv_buf = np.empty((), dtype=np.intc)
status = MPI.Status()

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
   comm_world.Ssend((snd_buf, 1, MPI.INT), dest=right, tag=17)
   # WRONG program, because synchronous MPI_Ssend will deadlock!
   comm_world.Recv((rcv_buf, 1, MPI.INT), source=left,  tag=17, status=status)
   np.copyto(snd_buf, rcv_buf) # We make a copy here. Remember that snd_buf = rcv_buf binds them to the same object (i.e. snd_buf and rcv_buf are then two different names pointing to the same object), which is unintended and can lead to incorrect code.
   sum += rcv_buf
print(f"PE{my_rank}:\tSum = {sum}")

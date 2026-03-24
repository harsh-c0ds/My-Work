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
#  Purpose: Substitute the ring algorithm by a collective proc. # 
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

snd_buf = np.array(my_rank, dtype=np.intc)

------ please substitute whole algorithm -----
right = (my_rank+1)      % size
left  = (my_rank-1+size) % size

sum = 0

for i in range(size):
   request = comm_world.Issend((snd_buf, 1, MPI.INT), dest=right,  tag=17)
   comm_world.Recv            ((rcv_buf, 1, MPI.INT), source=left, tag=17, status=status)
   request.Wait(status)
   np.copyto(snd_buf, rcv_buf) 
   sum += rcv_buf
------ by one call to a collective routine ---
------ input is my_rank, output is sum -------

----------------------------------------------

print(f"PE{my_rank}:\tSum = {sum}")

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
#  Purpose: Testing MPI_Neighbor_alltoall                       # 
#                                                               # 
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

rcv_buf_arr = np.empty((2),dtype=np.intc)
snd_buf_arr = np.empty((2),dtype=np.intc)
# snd_buf = snd_buf_arr[0];  rcv_buf = rcv_buf_arr[0] # ToDo ...

dims = np.empty(1,dtype=np.intc); periods = [False]
status = MPI.Status()

comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()

dims[0] = size; periods[0] = True; reorder = True
new_comm = comm_world.Create_cart(dims=dims, periods=periods, reorder=reorder)
my_rank = new_comm.Get_rank()
(left,right) = new_comm.Shift(0, 1)

sum = 0
snd_buf_arr[0] = my_rank # ToDo ...
# snd_buf_arr[?] = -1000-my_rank  # should be never used, only for test purpose # ToDo ...

for i in range(size):
   request = new_comm.Issend((snd_buf_arr[0:1], 1, MPI.INT), right, 17) # ToDo ...
   new_comm.Recv((rcv_buf_arr[0:1], 1, MPI.INT), left, 17, status) # ToDo ...
   request.Wait(status) # ToDo ...
   
   # _______.Neighbor_alltoall(_____________________________________________) # ToDo ...
   
   snd_buf_arr[0] = rcv_buf_arr[0] # ToDo ...
   sum += rcv_buf_arr[0]

print(f"PE{my_rank}:\tSum = {sum}")

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
#  Purpose: Creating a 1-dimens. topology with MPI_Cart_create  # 
#           & using MPI_Cart_shift to calculate neighbor ranks  # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################
  
from mpi4py import MPI
import numpy as np

rcv_buf = np.empty((),dtype=np.intc)
dims = np.empty(1,dtype=np.intc)
periods = [False]

status = MPI.Status()

# Get process info.
comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()

# Prepare input arguments for creating a Cartesian topology.
dims[0] = size
periods[0] = True
reorder = True

new_comm = comm_world.Create_cart(dims=dims, periods=periods, reorder=reorder)
# Get reordered my_rank (and coords if ndims>1)
my_rank = new_comm.Get_rank()
# my_coords = new_comm.Get_coords(my_rank)

# Get nearest neighbour rank.
(left,right) = new_comm.Shift(0, 1)

# The halo ring communication code from course chapter 4
sum = 0
snd_buf = np.array(my_rank, dtype=np.intc)

for i in range(size):
   request = new_comm.Issend((snd_buf, 1, MPI.INT), right, 17)
   new_comm.Recv((rcv_buf, 1, MPI.INT), left,  17, status)
   request.Wait(status)
   np.copyto(snd_buf, rcv_buf)
   sum += rcv_buf

print(f"PE{my_rank}:\tSum = {sum}")
# print(f"PE{my_rank}, Coords = {my_coords[0]}: Sum = {sum}")

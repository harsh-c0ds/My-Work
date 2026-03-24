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
#           Rolf Rabenseifner, Traugott Streicher (HLRS)        # 
#                                                               # 
#  Contact: rabenseifner@hlrs.de                                # 
#                                                               # 
#  Purpose: Creating a 2-dimensional topology.                  # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

max_dims = 2

rcv_buf = np.empty((),dtype=np.intc)

dims = [0]*max_dims
periods = [False]*max_dims
coords = np.empty((max_dims), dtype=np.intc)
remain_dims = [0]*max_dims

status = MPI.Status()

# Get process info.
comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()

# Set cartesian topology.
dims[0] = 0;          dims[1] = 0
periods[0] = True;    periods[1] = False
reorder = True

dims = MPI.Compute_dims(size, dims)
new_comm = comm_world.Create_cart(dims=dims, periods=periods, reorder=reorder)

# Get coords
my_rank = new_comm.Get_rank()
my_coords = new_comm.Get_coords(my_rank)

# Split the new_comm into slices
remain_dims[0]=1;  remain_dims[1]=0
slice_comm = new_comm.Sub(remain_dims)
size_of_slice = slice_comm.Get_size()
my_rank_in_slice = slice_comm.Get_rank()

# Get nearest neighbour rank.
(left,right) = slice_comm.Shift(0, 1)

# Compute global sum.
 
sum = 0;
snd_buf = np.array(my_rank, dtype=np.intc)

for i in range(size_of_slice):
   request = slice_comm.Issend((snd_buf, 1, MPI.INT), right, 17)

   slice_comm.Recv((rcv_buf, 1, MPI.INT), left,  17, status)

   request.Wait(status)
  
   np.copyto(snd_buf, rcv_buf)
   sum += rcv_buf;

print("PE{:2d}, Coords=({},{}), Slice_rank={}: Sum = {}".format( 
        my_rank, my_coords[0], my_coords[1], my_rank_in_slice, sum))

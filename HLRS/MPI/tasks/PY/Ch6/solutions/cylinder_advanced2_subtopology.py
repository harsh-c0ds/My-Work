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
#  Purpose: Creating a 2-dimensional topology.                  # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

to_right = 201
max_dims = 2

# Get process info.
comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()

# Set cartesian topology.
dims = [0, 0] # not necessary
periods = [1, 0]
reorder = True

dims = MPI.Compute_dims(size, max_dims)
new_comm = comm_world.Create_cart(dims, periods=periods, reorder=True)

# Get coords
my_rank = new_comm.Get_rank()

# Calls essentially MPI_Cart_coords(new_comm, my_rank, ndim, icoords) and returns icoords. ndim is derived using MPI_Cartdim_get(new_comm, &ndim).
my_coords = new_comm.Get_coords(my_rank)

# Split the new_comm into slices
remain_dims = [1, 0]
slice_comm = new_comm.Sub(remain_dims)
size_of_slice = slice_comm.Get_size()
my_rank_in_slice = slice_comm.Get_rank()

# Compute "global" sums for each slice
my_rank = np.array(my_rank, dtype=np.intc)
sum = np.empty((), dtype=np.intc)
slice_comm.Allreduce(my_rank, sum, MPI.SUM)

print(f"PE{my_rank}, Coords=({my_coords[0]},{my_coords[1]}), Slice_rank={my_rank_in_slice}: Sum = {sum}")

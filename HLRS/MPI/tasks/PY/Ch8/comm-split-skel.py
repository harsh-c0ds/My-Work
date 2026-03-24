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
#            Tobias Haas (HLRS)                                 # 
#                                                               # 
#  Contact: rabenseifner@hlrs.de                                # 
#                                                               # 
#  Purpose: A program to try MPI_Comm_split                     # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

sumA = np.empty((), dtype=np.intc)
sumB = np.empty((), dtype=np.intc)

comm_world = MPI.COMM_WORLD
world_size = comm_world.Get_size()
my_world_rank = np.array(comm_world.Get_rank(), dtype=np.intc)

# Compute sum of all ranks.
comm_world.Allreduce(my_world_rank, (sumA, 1, MPI.INT), MPI.SUM) 
comm_world.Allreduce(my_world_rank, (sumB, 1, MPI.INT), MPI.SUM)

print("PE world:{:3d}, color={:d} sub:{:3d}, SumA={:3d}, SumB={:3d} in comm_world".format( 
          my_world_rank, mycolor, my_sub_rank, sumA, sumB))

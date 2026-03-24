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
#  Purpose: A program to try MPI_Comm_create                    # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

sumA = np.empty((), dtype=np.intc)
sumB = np.empty((), dtype=np.intc)

ranges = np.empty((1,3), dtype=np.intc)

comm_world = MPI.COMM_WORLD
world_size = comm_world.Get_size()
my_world_rank = np.array(comm_world.Get_rank(), dtype=np.intc)

# This will make mycolor a boolean but the bool class is a subclass of int and behaves like int in numerical contexts (true=1, false=0).
mycolor = (my_world_rank > (world_size-1)//3)
# This definition of mycolor implies that the first color is 0

# instead of sub_comm = comm_world.Split(mycolor, 0)
#    ... the following code is used:

world_group = comm_world._________()

if (mycolor == 0):
   # first rank of my range:
   ranges[0][0] = 0
   # last  rank of my range:
   ranges[0][1] = (world_size-1)//3
else:
   # first rank of my range:
   ranges[0][0] = ___________
   # last  rank of my range:
   ranges[0][1] = __________
 
# stride of ranks:
ranges[0][2] = 1;

# print("PE world:{:3d}, color={:d} first={:d}, last={:d}, stride={:d}".format( 
#           my_world_rank, mycolor, ranges[0][0], ranges[0][1], ranges[0][2]))
  
________ = world_group.__________(______)
sub_comm = comm_world.______(_________)

sub_size = sub_comm.Get_size()
my_sub_rank = np.array(sub_comm.Get_rank(), dtype=np.intc)

# Compute sum of all ranks.
sub_comm.Allreduce (my_world_rank, (sumA, 1, MPI.INT), op=MPI.SUM) 
sub_comm.Allreduce (my_sub_rank,   (sumB, 1, MPI.INT), op=MPI.SUM) 

print("PE world:{:3d}, color={:d} sub:{:3d}, SumA={:3d}, SumB={:3d} in sub_comm".format( 
          my_world_rank, mycolor, my_sub_rank, sumA, sumB));

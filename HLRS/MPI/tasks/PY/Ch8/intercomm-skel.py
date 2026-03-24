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
#  Purpose: A program to try MPI_Intercomm_create               # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

sumA = np.empty((), dtype=np.intc)
sumB = np.empty((), dtype=np.intc)
sumC = np.empty((), dtype=np.intc)
sumD = np.empty((), dtype=np.intc)

comm_world = MPI.COMM_WORLD
world_size = comm_world.Get_size()
my_world_rank = np.array(comm_world.Get_rank(), dtype=np.intc)

mycolor = (my_world_rank > (world_size-1)//3)
# This definition of mycolor implies that the first color is 0
#  --> see calculation of remote_leader below
sub_comm = comm_world.Split(mycolor, 0)
sub_size = sub_comm.Get_size()
my_sub_rank = np.array(sub_comm.Get_rank(), dtype=np.intc)

# Compute sum of all ranks. */
sub_comm.Allreduce (my_world_rank, (sumA, 1, MPI.INT), op=MPI.SUM) 
sub_comm.Allreduce (my_sub_rank,   (sumB, 1, MPI.INT), op=MPI.SUM) 

# local leader in the lower group is locally rank 0
#    (to be provided in the lower group), 
#  and within MPI_COMM_WORLD (i.e., in peer_comm) rank 0
#(to be provided in the upper group) 
# local leader in the upper group is locally rank 0
#(to be provided in the upper group), 
#  and within MPI_COMM_WORLD (i.e., in peer_comm) 
#  rank 0+(size of lower group)
#(to be provided in the lower group)
if (mycolor==0): # This "if(...)" requires that mycolor==0 is the lower group!
   # lower group
   remote_leader = ______
else: # upper group
   remote_leader = ______

inter_comm = sub_comm.Create_intercomm(_____,comm_world,_____,_____)
my_inter_rank = np.array(inter_comm.Get_rank(), dtype=np.intc)

sub_comm.Allreduce  (my_inter_rank, (sumC, 1, MPI.INT), MPI.SUM) 
inter_comm.Allreduce(my_inter_rank, (sumD, 1, MPI.INT), MPI.SUM)

print("PE world:{0:3d}, color={1:d} sub:{2:3d} inter:{3:3d} SumA={4:3d}, SumB={5:3d}, SumC={6:3d}, SumD={7:3d}".format(my_world_rank, mycolor, my_sub_rank, my_inter_rank, sumA, sumB, sumC, sumD))

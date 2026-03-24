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
#  Purpose: Trying MPI_Scan in a ring topology.                 # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

tag_ready = 7781

sum = np.empty((),dtype=np.intc)
token = np.empty_like(sum)

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = np.array(comm_world.Get_rank(),dtype=np.intc)
size = comm_world.Get_size()

#  Compute partial rank sum.

comm_world.Scan(my_rank, (sum, 1, MPI.INT), op=MPI.SUM) 

# Output in natural order
# CAUTION: Although the printing is initialized by the 
#          MPI processes in the order of the ranks,
#          it is not guaranteed that the merge of the stdout
#          of the processes will keep this order
#          on the common output of all MPI processes !

if (my_rank != 0): 
   comm_world.Recv((token, 1, MPI.INT), source=my_rank-1, tag=tag_ready, status=status)


print(f"PE{my_rank}:\tSum = {sum}")
 
if (my_rank != size - 1):
   comm_world.Send((token, 1, MPI.INT), dest=my_rank + 1, tag=tag_ready)

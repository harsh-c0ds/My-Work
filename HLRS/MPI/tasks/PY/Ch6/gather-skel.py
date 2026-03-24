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
#  Authors: Joel Malard, Alan Simpson, (EPCC)                   # 
#           Rolf Rabenseifner, Tobias Haas (HLRS)               # 
#                                                               # 
#  Contact: rabenseifner@hlrs.de                                # 
#                                                               # 
#  Purpose: Gathering data from all processes                   # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np


#MPI-related data
comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

# doing some application work in each process, e.g.:
result = np.array(100.0 + 1.0 * my_rank, dtype=np.double)
print(f"I am process {my_rank} out of {num_procs}, result={result:f}")
if (my_rank == 0):
   result_array = np.empty(num_procs, dtype=np.double)

# ----------- the following gathering of the results should --------------
# -----------   be substituted by one call to MPI_Gather    --------------
if (my_rank != 0):  # in all processes, except "root" process 0
   # // sending some results from all processes (except 0) to process 0:
   comm_world.Send((result, 1, MPI.DOUBLE), dest=0, tag=99)
else:  # only in "root" process 0
   result_array[0] = result # process 0's own result
   # receiving all these messages
   for rank in range(1,num_procs):
      # result of processes 1, 2, ...
      comm_world.Recv((result_array[rank:], 1, MPI.DOUBLE), source=rank, tag=99, status=None) # we use slicing here to cut off the first elements
# ------------------ end of the gathering algorithm ----------------------

if (my_rank == 0):
   for rank in range(num_procs):
      print(f"I'm proc 0: result of process {rank} is {result_array[rank]:f}")

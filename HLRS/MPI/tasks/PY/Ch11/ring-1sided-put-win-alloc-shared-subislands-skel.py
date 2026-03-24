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
#  Purpose: A program to try out one-sided communication        #
#           with window=rcv_buf and MPI_PUT to put              #
#           local snd_buf value into remote window (rcv_buf).   #
#                                                               #
#  Contents: Python code, buffer send version (comm.Send)       #
#                                                               #
#################################################################

# ATTENTION!
# This code will work with mpi4py 3.0.0 and above, see comment below.

from mpi4py import MPI
import numpy as np

np_dtype = np.intc

comm_world = MPI.COMM_WORLD
my_rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

comm_sm = comm_world.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
my_rank_sm = comm_sm.Get_rank()
size_sm = comm_sm.Get_size()
if (my_rank_world == 0):
   if (size_sm == size_world):
      print("MPI_COMM_WORLD consists of only one shared memory region")
   else:
      print("MPI_COMM_WORLD is split into 2 or more shared memory islands")

split_method_SPLIT = True
# Before the first use of either of the both methods, you might have to build a helper module
# in subdirectory "helper". Run './build_ompi.sh' or './build_mpich.sh' in that directory
split_method_OPENMPI = False
split_method_MPICH = False
if split_method_SPLIT:
   # Splitting comm_sm into smaller sub-islands. Of course, they are also shared memory islands.
   _________________________________   # One may spilt also into more than 2 sub-islands
                                       # Rounding up with -1 / +1 trick
   _________________________________
   comm_sm_sub = comm_sm._________(_____, 0)
elif split_method_OPENMPI:
   # This split method is not defined by the MPI standard. Therefore the constants are not defined 
   # in the mpi4py module. We load a helper module (in subdirectory helper) to define these constants.
   import sys
   from pathlib import Path
   helper_path = (Path.cwd()).joinpath(Path('helper'))
   sys.path.append(str(helper_path))
   # Of course, one can spilt MPI_COMM_WORLD directly into its NUMA domains.
   # Here we split comm_sm into its NUMA domains.
   from split_helper_ompi.lib import OMPI_COMM_TYPE_NUMA
   comm_sm_sub = comm_sm.Split_type(OMPI_COMM_TYPE_NUMA, 0, MPI.INFO_NULL)
   # possible split types are: MPI_COMM_TYPE_SHARED,
   # OMPI_COMM_TYPE_NODE, OMPI_COMM_TYPE_HWTHREAD, OMPI_COMM_TYPE_CORE, OMPI_COMM_TYPE_L1CACHE,
   # OMPI_COMM_TYPE_L2CACHE, OMPI_COMM_TYPE_L3CACHE, OMPI_COMM_TYPE_SOCKET, OMPI_COMM_TYPE_NUMA,
   # OMPI_COMM_TYPE_BOARD, OMPI_COMM_TYPE_HOST, OMPI_COMM_TYPE_CU, OMPI_COMM_TYPE_CLUSTER
elif split_method_MPICH:
   # This split method is not defined by the MPI standard. Therefore the constants are not defined 
   # in the mpi4py module. We load a helper module (in subdirectory helper) to define these constants.
   import sys
   from pathlib import Path
   helper_path = (Path.cwd()).joinpath(Path('helper'))
   sys.path.append(str(helper_path))
   info = MPI.Info.Create()
   info.Set("SHMEM_INFO_KEY", "NUMA")  # This is not yet verified and tested :-(
   from split_helper_mpich.lib import MPIX_COMM_TYPE_NEIGHBORHOOD
   comm_sm_sub = comm_sm.Split_type(MPIX_COMM_TYPE_NEIGHBORHOOD, 0, info)

else:
   # no further splitting
   comm_sm_sub = comm_sm

my_rank_sm_sub = comm_sm_sub.Get_rank()
size_sm_sub = comm_sm_sub.Get_size()

right = (my_rank_sm_sub+1)             % size_sm_sub;
left  = (my_rank_sm_sub-1+size_sm_sub) % size_sm_sub;

# Allocate the window.
win = MPI.Win.Allocate_shared(np_dtype(0).itemsize*1, np_dtype(0).itemsize, MPI.INFO_NULL, comm_sm_sub)
# The buffer interface is not implemented for the Win classe prior to version 3.0.0. 
# This code will work with mpi4py 3.0.0 and above.
rcv_buf = np.frombuffer(win, dtype=np_dtype)
rcv_buf = np.reshape(rcv_buf,())

sum = 0
snd_buf = np.array(my_rank_sm_sub, dtype=np_dtype)

for i in range(size_sm_sub):
   win.Fence(MPI.MODE_NOSTORE | MPI.MODE_NOPRECEDE)
   win.Put((snd_buf, 1, MPI.INT), right, (0, 1, MPI.INT))
   win.Fence(MPI.MODE_NOSTORE | MPI.MODE_NOPUT | MPI.MODE_NOSUCCEED)

   np.copyto(snd_buf,rcv_buf)
   sum += rcv_buf

print("World: {} of {} comm_sm: {} of {} comm_sm_sub: {} of {} l/r={}/{} Sum = {}".format( 
      my_rank_world,size_world, my_rank_sm,size_sm, 
      my_rank_sm_sub,size_sm_sub, left,right,  sum))

win.Free()

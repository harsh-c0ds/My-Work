#!/usr/bin/env python3

#################################################################
#                                                               #
#  This file has been written as a sample solution to an        #
#  exercise in a course given at the High Performance           #
#  Computing Centre Stuttgart (HLRS).                           #
#  It is made freely available with the understanding that      #
#  every copy of this file must include this header and that    #
#  HLRS take no responsibility for the use of the               #
#  enclosed teaching material.                                  #
#                                                               #
#  Authors: Rolf Rabenseifner, Tobias Haas (HLRS)               #
#                                                               #
#  Contact: rabenseifner@hlrs.de                                #
#                                                               #
#  Purpose: Check version of the MPI library and include file   #
#                                                               #
#  Contents: Python code                                        #
#                                                               #
#################################################################

import sys
try:
   import mpi4py as mpi4py
   from mpi4py import MPI
except ModuleNotFoundError:
   print('Could not find mpi4py. This module is necessary for the python version. Please install or check our installation.')
   sys.exit()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()
(version, subversion) = MPI.Get_version()

try:
   import numpy as np
   numpy_version = np.__version__
   buffer = np.array([0], dtype='f') # Tests if this numpy routine works
except ModuleNotFoundError:
   if (my_rank == 0):
      print('WARNING: Could not find numpy. This module is necessary for better performance and most exercises. Please install or check your installation.')
   numpy_version = None

mpi4py_version = mpi4py.__version__

if (my_rank == 0):
   if (size > 1):
      print(f"Successful first MPI test executed in parallel on {size} processes using mpi4py version {mpi4py_version}.")
      if int(mpi4py_version[0]) < 3:
          print("CAUTION: You are using an mpi4py version below 3.0.0. We need a version above 3.0.0 in some exercises, so consider updating mpi4py.")
   else:
      print("CAUTION: This MPI test is executed only on one MPI process, i.e., sequentially!")
   print(f"Your installation supports MPI standard version {version}.{subversion}.")

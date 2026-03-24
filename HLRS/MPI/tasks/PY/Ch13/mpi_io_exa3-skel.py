#!/usr/bin/env python3

#########################################################################
#                                                                       #
#  This file has been written as a sample solution to an exercise in a  #
#  course given at the HLRS www.hlrs.de . It is made                    #
#  freely available with the understanding that every copy of this file #
#  must include this header and that HLRS takes no responsibility for   #
#  the use of the enclosed teaching material.                           #
#                                                                       #
#  Authors:    Rolf Rabenseifner, Tobias Haas                           #
#                                                                       #
#  Contact:    rabenseifner@hlrs.de                                     #
#                                                                       #
#  Purpose:    A program to test parallel file I/O with MPI.            #
#                                                                       #
#  Contents:   Python code, buffer send version (comm.Send)             #
#                                                                       #
#########################################################################

from mpi4py import MPI
import numpy as np

array_of_sizes = [None]
array_of_subsizes = [None]
array_of_starts = [None]
status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

etype = MPI.CHAR
array_of_sizes[0]    = size
array_of_subsizes[0] = 1
array_of_starts[0]   = my_rank
order = MPI.ORDER_C
filetype = etype.Create_subarray(array_of_sizes, array_of_subsizes,\
                                 array_of_starts, order)
filetype.Commit()

fh = MPI.File.Open(comm_world, "my_test_file", \
              MPI.MODE_RDWR | MPI.MODE_CREATE, \
              MPI.INFO_NULL)

disp = 0
fh.Set_view(disp, etype, filetype, "native", MPI.INFO_NULL)

for i in range(3):
   buf = np.array(ord('a')+my_rank, dtype=np.byte) # numpy.byte should be of integer type and compatible with C char
   fh.Write((buf, 1, etype), status)

fh.Close()

print(f"PE{my_rank}")

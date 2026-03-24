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
 
status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

fh = MPI.File.Open(comm_world, "my_test_file", \
              MPI.MODE_RDWR | MPI.MODE_CREATE, \
              MPI.INFO_NULL)

fh.Set_size(0) # removes the whole content if the file already existed

for i in range(10):
   buf = np.array(ord('0')+my_rank, dtype=np.byte) # numpy.byte should be of integer type and compatible with C char
   # Alternative: 
   # buf = np.array(str(my_rank), dtype=np.bytes_) # numpy.bytes_ represents a byte string 
   offset = my_rank + size*i;  
   fh.Write_at(offset, (buf, 1, MPI.CHAR), status)

fh.Close()

print(f"PE{my_rank}")

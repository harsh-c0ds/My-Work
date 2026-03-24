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
#  Purpose: A program to try MPI_Ssend and MPI_Recv.            # 
#                                                               # 
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI

buffer = [ None ]

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()

if (my_rank == 0):
   print(f"I am {my_rank} before send ping")
   comm_world.send(buffer, dest=1, tag=17);

elif (my_rank == 1):
   buffer = comm_world.recv(source=0, tag=17);
   print(f"I am {my_rank} after  recv ping ")

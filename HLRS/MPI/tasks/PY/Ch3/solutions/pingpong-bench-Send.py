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
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

number_of_messages = 50
buffer = np.array([0], dtype='f')
status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()

start = MPI.Wtime()
for i in range(1, number_of_messages+1):
   if (my_rank == 0):
      comm_world.Send((buffer,1,MPI.FLOAT), dest=1, tag=17)
      comm_world.Recv((buffer,1,MPI.FLOAT), source=1, tag=23, status=status)
   elif (my_rank == 1):
      comm_world.Recv((buffer,1,MPI.FLOAT), source=0, tag=17, status=status)
      comm_world.Send((buffer,1,MPI.FLOAT), dest=0, tag=23)

finish = MPI.Wtime()

if (my_rank == 0):
   msg_transfer_time = ((finish - start) / (2 * number_of_messages)) * 1e6 # in microsec
   print(f"Time for one messsage: {msg_transfer_time:f} micro seconds.")

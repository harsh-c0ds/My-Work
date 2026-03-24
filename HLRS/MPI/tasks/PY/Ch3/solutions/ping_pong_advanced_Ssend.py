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
#  Purpose: Benchmarking MPI_Ssend and MPI_Recv.                # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

process_A = 0
process_B = 1
ping  = 17
pong  = 23

number_of_messages = 50
start_length = 8
length_faktor = 64
max_length = 2097152     # 2 Mega
number_package_sizes = 4
status = MPI.Status()
buffer = np.zeros((max_length,), dtype='f')
# dtype='f' uses floats of size
size_of_float = buffer.itemsize

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()

if (my_rank == process_A):
   print("message size\t\ttransfertime\t\tbandwidth\n");

length_of_message = start_length

for i in range(1,number_package_sizes+1):
   if (my_rank == process_A):
      comm_world.Ssend((buffer, length_of_message, MPI.FLOAT), dest=process_B, tag=ping)
      comm_world.Recv((buffer, length_of_message, MPI.FLOAT), source=process_B, tag=pong, status=status)
   elif (my_rank == process_B):
      comm_world.Recv((buffer, length_of_message, MPI.FLOAT), source=process_A, tag=ping, status=status)
      comm_world.Ssend((buffer, length_of_message, MPI.FLOAT), dest=process_A, tag=pong)

   start = MPI.Wtime()
   for j in range(1,number_of_messages+1):
      if (my_rank == process_A):
         comm_world.Ssend((buffer, length_of_message, MPI.FLOAT), dest=process_B, tag=ping)
         comm_world.Recv((buffer, length_of_message, MPI.FLOAT), source=process_B, tag=pong, status=status)
      elif (my_rank == process_B):
         comm_world.Recv((buffer, length_of_message, MPI.FLOAT), source=process_A, tag=ping, status=status)
         comm_world.Ssend((buffer, length_of_message, MPI.FLOAT), dest=process_A, tag=pong)

   finish = MPI.Wtime()

   if (my_rank == process_A):
      time = finish - start
      transfer_time = time / (2 * number_of_messages)

      print(f"{length_of_message*size_of_float} bytes\t\t{transfer_time*1e6:f} usec\t\t{1.0e-6*length_of_message*size_of_float/transfer_time:f} MB/s")

   length_of_message = length_of_message*length_faktor

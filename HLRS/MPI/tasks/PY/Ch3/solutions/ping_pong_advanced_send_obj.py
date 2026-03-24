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
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import sys

process_A = 0
process_B = 1
ping  = 17
pong  = 23

number_of_messages = 50
start_length = 8
length_faktor = 64
max_length = 2097152     # 2 Mega
number_package_sizes = 4
# Assume size of float is 64 bit = 8 bytes
# This is depending on your interpreter!
size_of_float = 8
status = MPI.Status()
buffer = [0.0] * max_length

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
 
if (my_rank == process_A):
   print("message size\t\ttransfertime\t\tbandwidth\n");

length_of_message = start_length

for i in range(1,number_package_sizes+1):
   # Estimate actual size of object we will transfer
   msg_size = sys.getsizeof(buffer[0:length_of_message-1])
   if (my_rank == process_A):
      comm_world.send(buffer[0:length_of_message-1], dest=process_B, tag=ping)
      buffer[0:length_of_message-1] = comm_world.recv(source=process_B, tag=pong, status=status)
   elif (my_rank == process_B):
      buffer[0:length_of_message-1] = comm_world.recv(source=process_A, tag=ping, status=status)
      comm_world.send(buffer[0:length_of_message-1], dest=process_A, tag=pong)

   start = MPI.Wtime()
   for j in range(1,number_of_messages+1):
      if (my_rank == process_A):
         comm_world.send(buffer[0:length_of_message-1], dest=process_B, tag=ping)
         buffer[0:length_of_message-1] = comm_world.recv(source=process_B, tag=pong, status=status)
      elif (my_rank == process_B):
         buffer[0:length_of_message-1] = comm_world.recv(source=process_A, tag=ping, status=status)
         comm_world.send(buffer[0:length_of_message-1], dest=process_A, tag=pong)
         
   finish = MPI.Wtime()

   if (my_rank == process_A):
      time = finish - start
      transfer_time = time / (2 * number_of_messages)

      print(f"Brut: {msg_size} bytes\t\t{transfer_time*1e6:f} usec\t\t{1.0e-6*msg_size/transfer_time:f} MB/s")
      print(f"Net:  {size_of_float*length_of_message} bytes\t\t{transfer_time*1e6:f} usec\t\t{1.0e-6*size_of_float*length_of_message/transfer_time:f} MB/s\n")

   length_of_message = length_of_message*length_faktor

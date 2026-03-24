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
#  Authors: Rolf Rabenseifner, Tobias Haas (HLRS)               # 
#                                                               # 
#  Contact: rabenseifner@hlrs.de                                # 
#                                                               # 
#  Purpose: A program to try MPI_Send and MPI_Recv.             # 
#                                                               # 
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI

#   choose 10000 or 100000 to force that MPI_Send uses a synchronous protocol
#                          which will then cause a deadlock!
LNG = 100

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()

mass = [0.0] * LNG
temp = [0.0] * LNG

if (my_rank == 0):
   temp[0]=100;  mass[0]=2.5
   comm_world.send(temp, dest=1, tag=17)
   comm_world.send(mass, dest=1, tag=18)

elif (my_rank == 1):
   mass = comm_world.recv(source=0, tag=18, status=status)
   print(f"I am process {my_rank} and received mass={mass[0]:3.1f}")
   temp = comm_world.recv(source=0, tag=17, status=status)
   print(f"I am process {my_rank} and received temp={temp[0]:3.0f}")

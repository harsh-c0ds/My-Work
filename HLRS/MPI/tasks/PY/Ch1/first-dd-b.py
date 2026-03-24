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
#  Purpose: A first MPI example calculating the subdomain size  # 
#                                                               # 
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI

# 'Define' n, otherwise it can't be used in the broadcast below etc.
n = None

comm_world = MPI.COMM_WORLD
# MPI-related data
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

if (my_rank == 0):
   # reading the application data "n" from stdin only by process 0:
   n = int(input("Enter the number of elements (n): "))

# broadcasting the content of variable "n" in process 0 
# into variables "n" in all other processes:
n = comm_world.bcast(n, root=0)

# Calculating the number of elements of my subdomain: sub_n
# Calculating the start index sub_start within 0..n-1 
# or sub_start = -1 and sub_n = 0 if there is no element

# The following algorithm divides 5 into 2 + 1 + 1 + 1
sub_n = n // num_procs  # = floor(n/num_procs), i.e., rounding off
num_larger_procs = n - num_procs*sub_n  # = #procs with sub_n+1 elements
if (my_rank < num_larger_procs):
   sub_n = sub_n + 1
   sub_start = 0 + my_rank * sub_n
elif (sub_n > 0):
   sub_start = 0 + num_larger_procs + my_rank * sub_n
else:
   # this process has only zero elements
   sub_start = -1
   sub_n = 0

print(f"I am process {my_rank} out of {num_procs}, responsible for the {sub_n} elements with indexes {sub_start:{2}} .. {sub_start+sub_n-1:{2}}");

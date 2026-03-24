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
#  Purpose: Calculation of pi; load balancing tests             # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

def f(A):
   return 4.0/(1.0+A*A)

n = 10000000

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

# Calculating the number of elements of my subdomain: sub_n
# Calculating the start index sub_start within 0..n-1 
# or sub_start = -1 and sub_n = 0 if there is no element

# The following algorithm divides 5 into 2 + 1 + 1 + 1
sub_n = n // num_procs # = rounding_off(n/num_procs)
num_larger_procs = n - num_procs*sub_n # = #procs with sub_n+1 elements
if (my_rank < num_larger_procs):
   sub_n = sub_n + 1
   sub_start = 0 + my_rank * sub_n
elif (sub_n > 0):
   sub_start = 0 + num_larger_procs + my_rank * sub_n
else:
   # this process has only zero elements
   sub_start = -1
   sub_n = 0

if (num_procs >= 2):
   if (my_rank == num_procs-2 and sub_start >= 0):
      # taking all remaining iterations
      sub_n = n - sub_start
   if (my_rank == num_procs-1):
      # taking zero remaining iterations
      sub_start = -1; sub_n = 0

wt1=MPI.Wtime()

# calculate pi = integral [0..1] 4/(1+x**2) dx
w=1.0/n
p_sum=np.array(0.0, dtype=np.double)
sum=np.empty((),dtype=np.double)

for i in range(sub_start,sub_start+sub_n):
   x=w*(i+0.5)
   p_sum=p_sum+f(x)

wt2=MPI.Wtime()

comm_world.Reduce(p_sum, (sum, 1, MPI.DOUBLE), op=MPI.SUM, root=0)

print(f"PE{my_rank}/{num_procs}: sub_n={sub_n},  wall clock time = {wt2-wt1:12.4g} sec")

if (my_rank == 0):
   pi=w*sum
   print(f"PE{my_rank}/{num_procs}: computed pi = {pi:24.16g}")
   print(f"wall clock time = {wt2-wt1:12.4g} sec")

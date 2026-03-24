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

# EXERCISE:
# please add several MPI_Barrier statements and comment your reason for using it
comm_world.Barrier() # reason ___________
comm_world.Barrier() # reason ___________
comm_world.Barrier() # reason ___________
comm_world.Barrier() # reason ___________
# i.e., move these lines to useful locations, or add, or remove some

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

if (num_procs >= 3): # in principle, the following inbalance requires only at least 2 processes
                     # but "3" allows a balanced run with "mpirun -np 2"  :-)
   if (my_rank == num_procs-2 and sub_start >= 0):
      # taking all remaining iterations
      sub_n = n - sub_start
   if (my_rank == num_procs-1):
      # taking zero remaining iterations
      sub_start = -1; sub_n = 0


wt1=MPI.Wtime() # start time
 
# calculate pi = integral [0..1] 4/(1+x**2) dx
w=1.0/n
p_sum=np.array(0.0, dtype=np.double)
sum=np.empty((),dtype=np.double)

for i in range(sub_start,sub_start+sub_n):
   x=w*(i+0.5)
   p_sum=p_sum+f(x)

wt2=MPI.Wtime() # after numerics

wt3=MPI.Wtime() # before communication

comm_world.Reduce(p_sum, (sum, 1, MPI.DOUBLE), op=MPI.SUM, root=0)


wt4=MPI.Wtime() # end time
# EXERCISE:
#   Please substitute wt? by the appropriate wt1..wt4
# the following reduce statements are not critical to performance, therefore we are lazy and use lower-case reduce
wt_total     = wt? - wt?; wt_all_total   =   comm_world.reduce(wt_total, op=MPI.SUM, root=0) # root=0 and op=MPI.SUM are default, we omit them in the following
wt_numerics  = wt? - wt?; wt_all_numerics  = comm_world.reduce(wt_numerics)
wt_inbalance = wt? - wt?; wt_all_inbalance = comm_world.reduce(wt_inbalance)
wt_comm      = wt? - wt?; wt_all_comm   =    comm_world.reduce(wt_comm)

print(f"PE{my_rank}/{num_procs}: sub_n= {sub_n:7d}, {wt_numerics:12.4g} numerics +  {wt_inbalance:12.4g} inbalance" +
        f"+  {wt_comm:12.4g} comm = {wt_total:12.4g} sec in total")


if (my_rank == 0):
   pi=w*sum
   print(f"PE{my_rank}/{num_procs}: computed pi = {pi:24.16g}")
   print("average over all {:7d} processes: sub_n= {}, {:12.4g} numerics +  {:12.4g} inbalance +  {:12.4g} comm = {:12.4g} sec in total\n".format(
                      num_procs, n/num_procs, wt_all_numerics/num_procs, wt_all_inbalance/num_procs, wt_all_comm/num_procs, wt_all_total/num_procs))
   print("Parallel efficiency   = time in numeric  / total time = {:6.2f} %".format(wt_all_numerics /wt_all_total * 100))
   print("Loss by inbalance     = time in inbalance/ total time = {:6.2f} %".format(wt_all_inbalance/wt_all_total * 100))
   print("Loss by communication = time in comm     / total time = {:6.2f} %".format(wt_all_comm     /wt_all_total * 100)) 

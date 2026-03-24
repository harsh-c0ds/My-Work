#!/usr/bin/env python3
  
#################################################################
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
#            accelerated python with JIT compiler (numba)       # 
#                                                               # 
#################################################################

from numba import jit
from mpi4py import MPI
import numpy as np

# Uncomment one of the following two lines to use numba (JIT compiler)
# @jit(nopython=True,fastmath=True)
# @jit("float64(int64, float64, float64)", nopython=True,fastmath=True)
def f_sum(n, w, sum):
   for i in range(n):
      x=w*(i+0.5)
      sum=sum+4.0/(1.0+x*x)
   return sum

n = 10000000

wt1=MPI.Wtime()

# calculate pi = integral [0..1] 4/(1+x**2) dx
w=1.0/n
sum = 0.0
sum = f_sum(n,w, sum)

wt2=MPI.Wtime()

pi=w*sum;
print (f"computed pi = {pi:24.16g}")
print( f"wall clock time = {wt2-wt1:12.4g} sec")

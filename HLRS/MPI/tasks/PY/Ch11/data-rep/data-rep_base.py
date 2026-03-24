#!/usr/bin/env python3

##############################################################################
#                                                                            #
# data-replication in distributed and shared memory                          #
# program (C source code).                                                   #
#                                                                            #
# - the skeleton bcasts the data to all processes                            #
# - solution: rank_world == 0 puts the data                                  #
#             into the shared memory of node 0 ,                             #
#             rank_world == 0 bcasts the data to one of the processes        #
#             of each of the other nodes, only ,                             #
#             i.e., to all the other rank_shm==0 processes                   #
#                                                                            #
# - Course material: Introduction to Hybrid Programming in HPC               #
#                                                                            #
#                    It is made freely available with the understanding that #
#                    every copy must include this header and that            #
#                    the authors as well as VSC and TU Wien                  #
#                    take no responsibility for the use of this program.     #
#                                                                            #
#        (c) 01/2019 Irene Reichl (VSC Team, TU Wien)                        #
#                    irene.reichl@tuwien.ac.at                               #
#                                                                            #
#      vsc3:  module load intel/18 intel-mpi/2018                            #
#      							                     #
#                                                                            #
##############################################################################

from mpi4py import MPI
import numpy as np
# If numba installed, you can use numba. 
# Note that without numba the root process (rank==0) will temporarily use two 
# times the size of arrSize of memory!
nb_true = False

if nb_true:
   from numba import jit

arrType = np.int_
arrDataType = MPI.LONG # !!!!!   A C H T U N G :   MPI_Type an arrType anpassen         !!!!!
arrSize=int(16*1.6E7)

# ===> 1 <===

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

# ===> 2 <=== 
try:
   arr = np.empty(arrSize, dtype=arrType)
except MemoryError:
   print("arr NOT allocated, not enough memory")
   comm_world.Abort(0)

# ===> 3 <=== 

# If numba installed, we define a function for filling the array to use numba's jit
if nb_true:
   @jit("void(int64[:], int64, int64)", nopython=True)
   def fill(array, n, arrSize):
      for i in range(arrSize):
         array[i] = i + n
else:
   # No numba:
   np.copyto(arr, np.arange(-1,arrSize-1, dtype=arrType))

for it in range(3):
   # only rank_world=0 initializes the array arr
   if( rank_world == 0 ): 
      if nb_true:
         # We use a jit compiled function because pure python is slow here (try it).
         fill(arr,it,arrSize)
      else:
         arr += 1
      # If you want to try a pure python way, comment if/else above and uncomment
      # for i in range(arrSize):
      #    arr[i] = i + it
  
   # ===> 4 <=== 
   comm_world.Bcast( (arr, arrSize, arrDataType), 0)
   
   # Now, all arrays are filled with the same content.
   
   # ===> 5 <=== 
   # Use of compiled functions from np much faster than the for loop
   sum = np.sum(arr,dtype=np.longlong)
   #sum = int(0)
   #for i in range(arrSize):
   #   sum+= arr[ i ]
     
   # ===> 6 <=== 
   # To minimize the output, we print only from 3 process per SMP node        # TEST #
   if ( rank_world == 0 or rank_world == 1 or rank_world == size_world - 1 ): # TEST #
      print(f"it: {it}, rank ( world: {rank_world} ):\tsum(i={it}...i={arrSize-1+it}) = {sum:d}")

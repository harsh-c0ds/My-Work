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
#                                                                            #
#                                                                            #
##############################################################################

from mpi4py import MPI
import numpy as np
import sys
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
# TO DO
# comm_head; size_head, rank_head;

comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
size_world = comm_world.Get_size()

# ===> 2 <===   
#Create --> shared memory islands and --> shared memory window inside 
#          -->    comm_shm         and      -->    win                

comm_shm = comm_world.Split_type(MPI.COMM_TYPE_SHARED, key=0, info=MPI.INFO_NULL)
size_shm = comm_shm.Get_size()
rank_shm = comm_shm.Get_rank() 

# instead of:  arr = np.empty(arrSize, dtype=arrType)
if ( rank_shm == 0 ):
   individualShmSize = arrSize
else:
   individualShmSize = 0
win = MPI.Win.Allocate_shared( individualShmSize * arrType(0).itemsize, arrType(0).itemsize, MPI.INFO_NULL, comm_shm )
shm_buf_ptr = win.tomemory().address # This is actually only the address in memory, not a real pointer.
# pointer/memory from win object is not used because it is only available in process rank_shm==0

(buf, disp_unit) = win.Shared_query(0)
# This is only an assertion that all types match, not necessary.
assert disp_unit == arrDataType.Get_size()
assert disp_unit == arrType(0).itemsize
# To use the raw memory, we consider it as an numpy array
arr = np.frombuffer(buf, dtype=arrType)
arrSize_ = arr.nbytes

# To minimize the output, we print only from 3 process per SMP node # TEST #
if ( rank_shm == 0 or rank_shm == 1 or rank_shm == size_shm - 1 ):  # TEST #
   print("\t\trank ( world: {}, shm: {}) arrSize {} arrSize_ {}  shm_buf_ptr = 0x{:x}, arr_ptr = 0x{:x} ".format(
                    rank_world, rank_shm,    arrSize, arrSize_, shm_buf_ptr, buf.address )) 
if(rank_world==0):                                                  # TEST #
   print(f"ALL finalize and return !!!."); 

sys.exit(0) # Remeber sys.exit() still calles all exit handlers, i.e. also MPI.Finalize()

# TO DO: Create communicator comm_head with MPI_Comm_split  -->  including all the rank_shm == 0 processes.
# Only the rank_shm==0 processes should have a color, all others have color=MPI_UNDEFINED 
# this meands that only the head processes are in the communicator, the other processes are not included,
# i.e.,  their comm_head is MPI_COMM_NULL.
# The advantage of excluding all the other ranks compared to assigning color = 1 is that the communicator 
# is a smaller entity, hence, it allows for better scaling.
#
# Note that subsequent MPI-calls within comm_head must only be performed for processes on comm_head
# -->   if( comm_head != MPI_COMM_NULL ) 
#
# Choose key value in MPI_Comm_split_type(.... &comm_shm) and MPI_Comm_split(.... &comm_head) 
# such that the rank_world==0 process has also rank==0 within its comm_shm and within comm_head.


# color=MPI.UNDEFINED
# if (rank_shm==0):
#     color = 0
# comm_head = ...Split(... )

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

   # TO DO: the following line writes on the shared memory.
   #        take care for correct synchronization with memory fences !   
   #        The MPI_Win_fence starts the write epoch for all rank_shm==0 processes
   #
   # only rank_world=0 initializes the array arr
   if( rank_world == 0 ): # from those rank_shm=0 processes, only rank_world==0 fills arr
      if nb_true:
         # We use a jit compiled function because pure python is slow here (try it).
         fill(arr,it,arrSize)
      else:
         arr += 1
      # If you want to try a pure python way, comment if/else above and uncomment
      # for i in range(arrSize):
      #    arr[i] = i + it

  # ===> 4 <===
   # TO DO: do the Bcast only in comm_head !
   
   comm_world.Bcast( (arr, arrSize, arrDataType), 0)
   
   # Now, all arrays are filled with the same content.
   
   # ===> 5 <===
   # TO DO: the following lines read from the shared memory and compute the sum.
   #        take care for correct synchronization with memory fences !   
   #        The MPI_Win_fence now has to switch from the write epoch to the read epoch
   #        allowing reading arr by all processes belonging to their comm_shm.
   
   # Use of compiled functions from np much faster than the for loop
   sum = np.sum(arr,dtype=np.longlong)
   #sum = int(0)
   #for i in range(arrSize):
   #   sum+= arr[ i ]
   
   # ===> 6 <=== 
   print(f"it: {it}, rank ( world: {rank_world} ):\tsum(i={it}...i={arrSize-1+it}) = {sum:d}")
   # already prepared for the shared memory solution:
   # print(f"it: {it}, rank ( world: {rank_world}, shm: {rank_shm}, head: {rank_head} ):\tsum(i={it}...i={arrSize-1+it}) = {sum}")

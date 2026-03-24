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
# If numba installed, you can use numba. 
# Note that the root process (rank==0) will temporarily use two times the size
# of arrSize of memory!
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

if ( rank_shm == 0 or rank_shm == 1 or rank_shm == size_shm - 1 ):
   print("\t\tprocess {}   arrSize {} arrSize_ {}  shm_buf_ptr = 0x{:x}, arr_ptr = 0x{:x} ".format(
                rank_world, arrSize, arrSize_, shm_buf_ptr, buf.address )) # instead of buf.address we could also use arr.__array_interface__['data'][0], the numpy array's buffer address, which is the same.

# Create communicator including all the rank_shm = 0
# with the MPI_Comm_split: in color 0 all the rank_shm = 0 ,
# all other ranks are color = 1

color=MPI.UNDEFINED
if (rank_shm==0):
   color = 0

comm_head = comm_world.Split(color, key=0)
rank_head = -1; # only used in the print statements to differentiate unused rank==-1 from used rank==0
if( comm_head != MPI.COMM_NULL ): # if( color == 0 ) // rank is element of comm_head, i.e., it is head of one of the islands in comm_shm
   size_head = comm_head.Get_size()
   rank_head = comm_head.Get_rank()

# ADD ON: calculates the minimum and maximum size of size_shm
mm = np.array([-size_shm, size_shm], dtype=np.intc)
minmax = np.empty_like(mm)

if( comm_head != MPI.COMM_NULL ):
   comm_head.Reduce( (mm,2,MPI.INT), (minmax, 2, MPI.INT), MPI.MAX, 0)
   
   if( rank_world == 0 ):
      print(f"\n\tThe number of shared memory islands is: {size_head} islands ")
      if ( minmax[0] + minmax[1] == 0 ):
         print(f"\tThe size of all shared memory islands is: {-minmax[0]} processes")
      else:
         print("\tThe size of the shared memory islands is between min = {-minmax[0]} and max = {minmax[1]} processes ")

# End of ADD ON. Note that the following algorithm does not require same sizes of the shared memory islands 

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
   # all rank_shm=0 start the write epoch: writing arr to their shm
   win.Fence( 0 ) # workaround: no assertions 
   if( rank_world == 0 ): # from those rank_shm=0 processes, only rank_world==0 fills arr
       if nb_true:
          # We use a jit compiled function because pure python is slow here (try it).
          fill(arr,it,arrSize)
       else:
          arr += 1
      # If you want to try a pure python way, comment if/else above and uncomment
      #for i in range(arrSize):
      #   arr[i] = i + it

   # ===> 4 <===
   # Instead of all processes in MPI_COMM_WORLD, now only the heads of the 
   # shared memory islands communicate (using comm_head).
   # Since we used key=0 in both MPI_Comm_split(...), process rank_world = 0
   # - is also rank 0 in comm_head
   # - and rank 0 in comm_shm in the color it belongs to.
   
   if( comm_head != MPI.COMM_NULL ): # if( color == 0 )
      comm_head.Bcast((arr, arrSize, arrDataType), 0)
      # with this Bcast, all other rank_shm=0 processes write the data into their arr
   
   # Now, all arrays are filled with the same content.
   
   # ===> 5 <===
   win.Fence( 0 ) # workaround: no assertions; after the fence all processes start a read epoch
   
   # Now, all other ranks in the comm_sm shared memory islands are allowed to access their shared memory array.
   # And all ranks rank_sm access the shared mem in order to compute sum

   # Use of compiled functions from np much faster than the for loop
   sum = np.sum(arr,dtype=np.longlong)
   #sum = int(0)
   #for i in range(arrSize):
      # sum+= *( shm_buf_ptr - rank_shm * shmSize + i )
   #  sum+= arr[ i ]
     
   # ===> 6 <===
   # To minimize the output, we print only from 3 process per SMP node # TEST #
   if ( rank_shm == 0 or rank_shm == 1 or rank_shm == size_shm - 1 ):  # TEST #
      print("it: {}, rank ( world: {}, shm: {}, head: {} ):\tsum(i={}...i={}) = {} ".format(
                      it, rank_world, rank_shm, rank_head, it, arrSize-1+it, sum ))
   # end of it-loop

# ===> 7 <===
win.Fence(0) # workaround: no assertions; free destroys the shm. fence to guarantee that read epoch has been finished
# It is not necessary, since we are at the end of the program, but for safety we bind the names arr and buf to None, since win.Free() deallocates the underlying buffer. 
# In C we would move the pointers to NULL.
arr = buf = None
# Note that the __dealloc__ function of the Win class calls MPI_Win_free (if the win is not NULL or MPI_WIN_NULL), so it would not be necessary at the end of the program.
win.Free()

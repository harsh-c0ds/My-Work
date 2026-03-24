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
#  Purpose: Trying to measure whether progress may require      #
#           the call of an unspecific MPI routines in another   #
#           process                                             #
#                                                               #
#  One may modify the message count (change BUF_CNT_SIZE=10000) #
#                                                               #
#  Contents: Python code, buffer send version (comm.Send)       #
#                                                               #
#################################################################

BUF_CNT_SIZE = 1000000

from mpi4py import MPI
import numpy as np
from time import sleep

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

# To be sure that all processes are started.
comm_world.Barrier() 
sleep(1) # to be sure that all process left the MPI_Barrier

if (my_rank == 0): # buffered sender
   print("\nMessage size is %ld bytes \n".format(BUF_CNT_SIZE * np.single(0).size))

   snd_buf = np.empty(BUF_CNT_SIZE, dtype=np.single)
   bsend_buffer_size = MPI.FLOAT.Pack_size(BUF_CNT_SIZE, comm_world)
   bsend_buffer_size = bsend_buffer_size + MPI.BSEND_OVERHEAD
   bsend_buffer = np.empty(bsend_buffer_size, dtype=np.byte)
   MPI.Attach_buffer(bsend_buffer)

   print("I am {} before bsend call ".format(my_rank))
   time_begin = MPI.Wtime()
   comm_world.Bsend((snd_buf, BUF_CNT_SIZE, MPI.FLOAT), 1, 17)
   time_end = MPI.Wtime()
   print("I am {} after bsend call, delta = time {:9.6f} sec ".format(my_rank, time_end-time_begin ))
   print("I am {} before sleep(40) ".format(my_rank))
   time_begin = MPI.Wtime()
   sleep(40)
   time_end = MPI.Wtime()
   print("I am {} after sleep(40), delta = time {:9.6f} sec ".format(my_rank, time_end-time_begin ))
   print("I am {} before barrier call ".format(my_rank))
   time_begin = MPI.Wtime()
   comm_world.Barrier() # as an unspecific MPI call that allows to complete the internal work for the (weak) local MPI_Bsend
   time_end = MPI.Wtime()
   print("I am {} after barrier call, delta = time {:9.6f} sec ".format(my_rank, time_end-time_begin ))

elif (my_rank == 1): # delayed receiver
   rcv_buf = np.empty(BUF_CNT_SIZE, dtype=np.single) 
 
   print("                              I am {} before sleep(15) ".format(my_rank))
   time_begin = MPI.Wtime()
   sleep(15) # 15 seconds ago, process 0 has Bsend the message
   time_end = MPI.Wtime()
   print("                              I am {} after sleep(15), delta = time {:9.6f} sec ".format(my_rank, time_end-time_begin ))
   print("                              I am {}: 15 seconds ago, process 0 called MPI_Bsend ".format(my_rank ))
   print("                              I am {} before recv call ".format(my_rank))
   print("                              I am {}: With progress, the following MPI_Recv will come back after a few micro seconds ".format(my_rank ))
   print("                              I am {}: Without progress, the following MPI_Recv will wait another 25 seconds until process 0 will call the barrier = unspecific call following the Bsend ".format(my_rank ))
   time_begin = MPI.Wtime()
   comm_world.Recv((rcv_buf, BUF_CNT_SIZE, MPI.FLOAT), 0, 17)
   time_end = MPI.Wtime()
   print("                              I am {} after recv call, delta time = {:9.6f} sec ".format(my_rank, time_end-time_begin ))
   print("                              I am {} before barrier call ".format(my_rank))
   print("                              I am {}: With progress, the previous MPI_Recv as instantly done and the barrier has to wait for the other 25 sec ".format(my_rank ))
   print("                              I am {}: Without progress, the Recv waited for the other 25 sec and therefore the following barrier needs only a few micro seconds ".format(my_rank ))
   time_begin = MPI.Wtime()
   comm_world.Barrier() # as an unspecific MPI call that allows to complete the internal work for the (weak) local MPI_Bsend in the other process
   time_end = MPI.Wtime()
   print("                              I am {} after barrier call, delta = time {:9.6f} sec ".format(my_rank, time_end-time_begin ))

else: # other processes that are not involved
   comm_world.Barrier() # that all processes have called the barrier

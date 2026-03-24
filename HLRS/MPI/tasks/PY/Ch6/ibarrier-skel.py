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
#  Purpose: A program to try MPI_Ibarrier.                      # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

# in the role as sending process
snd_buf_A = np.empty((), dtype=np.intc); snd_buf_B = np.empty((), dtype=np.intc)
snd_buf_C = np.empty((), dtype=np.intc); snd_buf_D = np.empty((), dtype=np.intc)
dest = number_of_dests = 0
snd_finished=0
snd_rq = [MPI.Request()]*4      # Not strictly necessary, since we will assigne other objects later. We could also simply append to an empty list.
total_number_of_dests  = 0 # only for verification, should be removed in real applications (intialization here not necessary in python)
                           # Caution: total_number_of_dests may be less than 4, see if-statements below
# in the role as receiving process
rcv_buf = np.empty((), dtype=np.intc)
ib_finished=False
rcv_sts = MPI.Status()
number_of_recvs = total_number_of_recvs = 0  # only for verification, should be removed in real applications

round=0 # only for verification, should be removed in real applications

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

# in the role as sending process
dest = my_rank+1;
if ((dest>=0) and (dest<size)):
   snd_buf_A = np.array(1000*my_rank + dest, dtype=np.intc)  # must not be modified until send-completion with TEST or WAIT
   snd_rq[number_of_dests] = comm_world.Issend((snd_buf_A,1,MPI.INT), dest=dest,tag=222)
   print("A rank: {0:3d} - sending_: message {1:06d} from {0:3d} to {2:3d}".format(my_rank,snd_buf_A,dest))
   number_of_dests += 1

dest = my_rank-2
if ((dest>=0) and (dest<size)):
   snd_buf_B = np.array(1000*my_rank + dest, dtype=np.intc)   # must not be modified until send-completion with TEST or WAIT
   snd_rq[number_of_dests] = comm_world.Issend((snd_buf_B,1,MPI.INT), dest=dest,tag=222)
   print("A rank: {0:3d} - sending_: message {1:06d} from {0:3d} to {2:3d}".format(my_rank,snd_buf_B,dest))
   number_of_dests += 1

dest = my_rank+4
if ((dest>=0) and (dest<size)):
   snd_buf_C = np.array(1000*my_rank + dest, dtype=np.intc)  # must not be modified until send-completion with TEST or WAIT
   snd_rq[number_of_dests] = comm_world.Issend((snd_buf_C,1,MPI.INT), dest=dest,tag=222)
   print("A rank: {0:3d} - sending_: message {1:06d} from {0:3d} to {2:3d}".format(my_rank,snd_buf_C,dest))
   number_of_dests += 1

dest = my_rank-7
if ((dest>=0) and (dest<size)):
   snd_buf_D = np.array(1000*my_rank + dest, dtype=np.intc)  # must not be modified until send-completion with TEST or WAIT */
   snd_rq[number_of_dests] = comm_world.Issend((snd_buf_D,1,MPI.INT), dest=dest,tag=222)
   print("A rank: {0:3d} - sending_: message {1:06d} from {0:3d} to {2:3d}".format(my_rank,snd_buf_D,dest))
   number_of_dests += 1
 
while( not ib_finished):
   # in the role as receiving process 
   #  MPI_IPROBE(MPI_ANY_SOURCE); If there is a message then MPI_RECV for this one message:
   rcv_flag = comm_world.Iprobe(______________,___, _________)
   if(rcv_flag):
      comm_world.Recv((rcv_buf,1,MPI.INT), source=MPI.ANY_SOURCE,tag=222, status=rcv_sts)
      print("A rank: {0:3d} - received: message {1:06d} from {2:3d} to {0:3d}".format(my_rank,rcv_buf,rcv_sts.Get_source()))
      number_of_recvs += 1 # only for verification, should be removed in real applications
      
   # in the role as sending process
   if(not snd_finished): # the following lines make only sense as long as not all MPI_ISSENDs are finished
      # Check whether all MPI_ISSENDs are finished
      snd_finished = MPI.Request.Te_____( ______, ____________ ) # statuses=None is default
      # if all MPI_ISSENDs are finished then call MPI_IBARRIER
      if(____________): # i.e., the first time, i.e., only once
        ib_rq = ________.________()
      
   # loop until MPI_IBARRIER finished (i.e. all processes signaled that all receives are called)
   if(____________):  # the test whether the MPI_IBARRIER is finished
                      #  can be done only if MPI_IBARRIER is already started.
                      #  This ist true as soon snd_finished is true */
      ___________ = __________(status=None) 

#  only for verification, should be removed in real applications, therefore call of lower-case version okay:
total_number_of_dests = comm_world.reduce(number_of_dests, op=MPI.SUM, root=0)
total_number_of_recvs = comm_world.reduce(number_of_recvs, MPI.SUM, 0)        # here with positional arguments
if (my_rank == 0):
   print("B #sends= {0:5d}  /  #receives= {1:5d}".format(total_number_of_dests,total_number_of_recvs))  
   if (total_number_of_dests != total_number_of_recvs):
       print("ERROR !!!! Wrong number of receives")

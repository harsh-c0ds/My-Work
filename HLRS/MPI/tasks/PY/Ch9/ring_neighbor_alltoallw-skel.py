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
#  Purpose: Testing MPI_Neighbor_alltoallw                      # 
#  Purpose: Using MPI_Neighbor_alltoallw for ring communication.# 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

dims = np.empty(1,dtype=np.intc); periods = [False]

snd_displs = [None]*2; rcv_displs = [None]*2
snd_counts = [None]*2; rcv_counts = [None]*2
snd_types = [None]*2; rcv_types = [None]*2

status = MPI.Status()

comm_world = MPI.COMM_WORLD
size = comm_world.Get_size()
 
dims[0] = size
periods[0] = True
reorder = True

new_comm = comm_world.Create_cart(dims=dims, periods=periods, reorder=reorder)
my_rank = new_comm.Get_rank()
(left,right) = new_comm.Shift(0, 1)

sum = 0
snd_buf = np.array(my_rank, dtype=np.intc)
rcv_buf = np.array(-1000, dtype=np.intc) # unused value, should be overwritten by first MPI_Recv; only for test purpose
# In contrast to C or Fortran, mpi4py needs objects implementing the Python buffer protocol.
# We cannot simply use MPI.BOTTOM as base addresse (for neither sendbuf nor recvbuf) 
# but we can use a 'bottom buffer' to pass MPI.BOTTOM as address.
mem_from_bottom = MPI.memory.fromaddress(MPI.BOTTOM,0,0)

rcv_counts[0] = 1
rcv_displs[0] = MPI.Get_address(rcv_buf)
rcv_types[0] = MPI.INT
rcv_counts[1] = 0
rcv_displs[1] = 0 # unused
rcv_types[1] = MPI.INT
snd_counts[0] = _____
_____ = _____
_____ = _____
snd_counts[1] = _____
______ = _____
______ = _____

for i in range(size):
   # Substituted by MPI_Neighbor_alltoallw() :
   # MPI_Issend(&snd_buf, 1, MPI_INT, right, 17, new_comm, &request);
   # MPI_Recv(&rcv_buf, 1, MPI_INT, left, 17, new_comm, &status);
   # MPI_Wait(&request, &status);
    
   new_comm.Neighbor_alltoallw((__________, snd_counts, snd_displs, snd_types), 
                               (__________, rcv_counts, rcv_displs, rcv_types))

   np.copyto(snd_buf, rcv_buf)
   sum += rcv_buf

print(f"PE{my_rank}:\tSum = {sum}")


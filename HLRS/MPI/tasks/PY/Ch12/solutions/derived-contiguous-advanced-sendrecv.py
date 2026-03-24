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
#  Purpose: A program to try MPI_Sendrecv.                      #
#                                                               #
#  Contents: Python code, buffer send version (comm.Send)       #
#                                                               #
#################################################################

from mpi4py import MPI
import numpy as np

np_dtype = np.dtype([('i', np.intc), ('j', np.intc)])

snd_buf = np.empty((),dtype=np_dtype)
rcv_buf = np.empty_like(snd_buf)
sum = np.empty_like(snd_buf)

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

right = (my_rank+1)      % size
left  = (my_rank-1+size) % size

# Set MPI datatypes for sending and receiving partial sums.
send_recv_type = MPI.INT.Create_contiguous(2)
send_recv_type.Commit()

sum['i'] = 0;            sum['j'] = 0
snd_buf['i'] = my_rank;  snd_buf['j'] = 10*my_rank  # Step 1 = init

for i in range(size):
   comm_world.Sendrecv((snd_buf, 1, send_recv_type), right, 17,  # Step 2
                       (rcv_buf, 1, send_recv_type), left, 17,   # Step 3
                       status)
   np.copyto(snd_buf,rcv_buf)                          # Step 4
   sum['i'] += rcv_buf['i'];  sum['j'] += rcv_buf['j'] # Step 5

print(f"PE{my_rank}:\tSum = {sum['i']}\t{sum['j']}")



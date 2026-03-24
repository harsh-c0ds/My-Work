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
#  Purpose: A program with derived datatypes.                   #
#                                                               #
#  Contents: Python code, buffer send version (comm.Send)       #
#                                                               #
#################################################################

from mpi4py import MPI
import numpy as np

np_dtype = np.dtype([('i', np.intc), ('f', np.single)])
snd_buf = np.empty((),dtype=np_dtype)
rcv_buf = np.empty_like(snd_buf)
sum = np.empty_like(snd_buf)

array_of_blocklengths = [None]*2
array_of_displacements = [None]*2
array_of_types = [None]*2

status = MPI.Status()

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

right = (my_rank+1)      % size
left  = (my_rank-1+size) % size

# Set MPI datatypes for sending and receiving partial sums.
array_of_blocklengths[0] = 1
array_of_blocklengths[1] = 1

first_var_address = MPI.Get_address(snd_buf['i'])
second_var_address = MPI.Get_address(snd_buf['f'])

array_of_displacements[0] = 0
array_of_displacements[1] = MPI.Aint_diff(second_var_address, first_var_address)

array_of_types[0] = MPI.INT
array_of_types[1] = MPI.FLOAT

send_recv_type = MPI.Datatype.Create_struct(array_of_blocklengths, array_of_displacements, array_of_types)
send_recv_type.Commit()

sum['i'] = 0;            sum['f'] = 0
snd_buf['i'] = my_rank;  snd_buf['f'] = 10*my_rank  # Step 1 = init

for i in range(size):
   request = comm_world.Issend((snd_buf, 1, send_recv_type), dest=right, tag=17)      # Step 2a
   comm_world.Recv((rcv_buf, 1, send_recv_type), source=left, tag=17, status=status)  # Step 3
   request.Wait(status)                                                               # Step 2b
   np.copyto(snd_buf,rcv_buf)                          # Step 4
   sum['i'] += rcv_buf['i'];  sum['f'] += rcv_buf['f'] # Step 5

print(f"PE{my_rank}:\tSum = {sum['i']}\t{sum['f']}")

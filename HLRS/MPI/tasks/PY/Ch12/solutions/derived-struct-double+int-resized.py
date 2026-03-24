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
#           Rolf Rabenseifner, Traugott Streicher (HLRS)        #
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

arr_lng = 5

# What happens if align=True/False? Why?
np_dtype = np.dtype([('f', np.double), ('i', np.intc)], align=True)
snd_buf = np.empty(arr_lng,dtype=np_dtype)
rcv_buf = np.empty_like(snd_buf)
sum = np.empty_like(snd_buf)

array_of_blocklengths = [None]*2
array_of_displacements = [None]*2
array_of_types = [None]*2

status = MPI.Status()

# Get process and neighbour info.
comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
size = comm_world.Get_size()

right = (my_rank+1)      % size
left  = (my_rank-1+size) % size

# Set MPI datatypes for sending and receiving partial sums.
array_of_blocklengths[0] = 1
array_of_blocklengths[1] = 1

array_of_displacements[0] = 0
# dtype.fields is a tuple of (type, offset)
array_of_displacements[1] = np_dtype.fields['i'][1]

array_of_types[0] = MPI.DOUBLE
array_of_types[1] = MPI.INT

send_recv_type = MPI.Datatype.Create_struct(array_of_blocklengths, array_of_displacements, array_of_types)
send_recv_resized = send_recv_type.Create_resized(0, snd_buf.itemsize)
send_recv_resized.Commit()

# Compute global sum.
for j in range(arr_lng):
   sum[j]['i'] = 0;            sum[j]['f'] = 0
   snd_buf[j]['i'] = (j+1)*my_rank;  snd_buf[j]['f'] = (j+1)*my_rank  # Step 1 = init
   rcv_buf[j]['i'] = -1;  rcv_buf[j]['f'] = -1

for i in range(size):
   # to check, whether the data transfer is correct, we do not transfer the last index
   request = comm_world.Issend((snd_buf, arr_lng-1, send_recv_resized), right, 17)  # Step 2a
   comm_world.Recv((rcv_buf, arr_lng-1, send_recv_resized), left, 17, status);      # Step 3
   request.Wait(status)                                                          # Step 2b
   for j in range(arr_lng):
      snd_buf[j] = rcv_buf[j]                                          # Step 4
      sum[j]['i'] += rcv_buf[j]['i'];  sum[j]['f'] += rcv_buf[j]['f']  # Step 5

if (my_rank==0):
   buf_mpi_size = send_recv_resized.Get_size()
   (buf_mpi_lb, buf_mpi_extent) = send_recv_resized.Get_extent()
   (buf_mpi_lb, buf_mpi_true_extent) = send_recv_resized.Get_true_extent()
   print("A-- MPI_Type_size:            {:3d}".format(buf_mpi_size))
   print("B-- MPI_Type_get_true_extent: {:3d}".format(buf_mpi_true_extent))
   print("C-- MPI_Type_get_extent:      {:3d}".format(buf_mpi_extent))
   buf_mpi_size = send_recv_resized.Get_size()
   (buf_mpi_lb, buf_mpi_extent) = send_recv_resized.Get_extent()
   (buf_mpi_lb, buf_mpi_true_extent) = send_recv_resized.Get_true_extent()
   print("D-- send_recv_resized:")
   print("E-- MPI_Type_size:            {:3d}".format(buf_mpi_size))
   print("F-- MPI_Type_get_true_extent: {:3d}".format(buf_mpi_true_extent))
   print("G-- MPI_Type_get_extent:      {:3d}".format(buf_mpi_extent))
   print("H-- sizeof:                   {:3d}".format(snd_buf.itemsize))
   mem = MPI.memory.frombuffer(sum) 
   print("I-- real size is:             {:3d}".format(int(mem.nbytes/sum.size)))
   if (buf_mpi_extent != int(mem.nbytes/sum.size)):
      print("J--  CAUTION:  mismatch of language type and MPI derived type: {:3d} != {:3d}".format(
             buf_mpi_extent, int(mem.nbytes/sum.size)))

   print("K--")
   print("L-- Expected results: for all, except the highjest j:  sum = (j+1)*(sum of all ranks)")
   print("M-- For the highest j value, no data exchange is done: sum = -(number of processes)")

for j in range(arr_lng):
   print("PE{:3d} j={:3d}: Sum = {:6d}  {:8.1f}".format(my_rank, j, sum[j]['i'], sum[j]['f']))

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
#  Purpose: A program to try MPI_Issend and MPI_Recv.           # 
#                                                               # 
#  Contents: Python code, buffer send version (comm.Send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI
import numpy as np

to_right = 201

dims = np.empty(2, dtype=np.intc)
periods = [None, None]
rcv_buf = np.empty((), dtype=np.intc)
status = MPI.Status()

comm_world = MPI.COMM_WORLD
world_size = comm_world.Get_size()
my_world_rank = comm_world.Get_rank()

# Large example with 2*3 cores * 2*2 CPUs * 3*3 nodes (see slides):
inner_d0=2; mid_d0=2; outer_d0=3;
inner_d1=3; mid_d1=2; outer_d1=3;
# A small example with 2*1 cores * 2*2 CPUs * 3*1 nodes:
inner_d0=2; mid_d0=2; outer_d0=3;
inner_d1=1; mid_d1=2; outer_d1=1;
# Small example with 2*2 cores * 2*1 CPUs * 1*3 nodes (see slides):
inner_d0=2; mid_d0=2; outer_d0=1;
inner_d1=2; mid_d1=1; outer_d1=3;
dim0=inner_d0*mid_d0*outer_d0; 
dim1=inner_d1*mid_d1*outer_d1; 
idim=inner_d0*inner_d1; mdim=mid_d0*mid_d1; 
odim=outer_d0*outer_d1;
whole_size=dim0*dim1;
ranks = np.empty(whole_size, dtype=np.intc);
for oc0 in range(outer_d0): # any sequence of the nested loop works
   for mc0 in range(mid_d0):
      for ic0 in range(inner_d0):
         for oc1 in range(outer_d1):
           for mc1 in range(mid_d1):
              for ic1 in range(inner_d1):
                 old_rank = ic1 + inner_d1*ic0 \
                  + (mc1 + mid_d1  *mc0)*idim \
                  + (oc1 + outer_d1*oc0)*idim*mdim
                 c0 = ic0 + inner_d0*mc0 + inner_d0*mid_d0*oc0
                 c1 = ic1 + inner_d1*mc1 + inner_d1*mid_d1*oc1
                 new_rank = c1 + dim1*c0     
                 ranks[new_rank] = old_rank

# only for debug-print:
if (my_world_rank==0):
   for new_rank in range(whole_size):
      print("new= {:3d} old= {:3d}".format(new_rank, ranks[new_rank]))
   print("")
   print("     c1=", end = '')
   for c1 in range(dim1):
      print(" {:3d}".format(c1), end = '')
   print("")
   print("old_rank", end = '')
   for c1 in range(dim1):
      print("----", end = '')
   print("")
   for c0 in range(dim0):
      print("c0= {:2d} |".format(c0), end = '')
      for c1 in range(dim1): 
         print(" {:3d}".format(ranks[c1+dim1*c0]), end = '')  
      print("")
   print("")
   print("     c1=", end = '')
   for c1 in range(dim1):
      print(" {:3d}".format(c1), end = '')
   print("")
   print("new_rank", end = '')
   for c1 in range(dim1):
      print("----", end = '')
   print("")
   for c0 in range(dim0):
      print("c0= {:2d} |".format(c0), end = '')
      for c1 in range(dim1):
         print(" {:3d}".format(c1+dim1*c0), end = '') 
      print("")
   print("")

if (whole_size==world_size):  
   # Establishing new_comm with the new ranking in a array "ranks":
   world_group = comm_world.Get_group()
   new_group = world_group.Incl(ranks)
   new_comm = comm_world.Create(new_group)

   # testing the new communicator, e.g., with our ring algorithm:
   new_size = new_comm.Get_size() # should be the original one
   my_new_rank = new_comm.Get_rank()
   # Source code without Cartesian Virtual Toplogy
   c0 = my_new_rank // dim1      # with 3 dims: c0 = my_new_rank / (dim1*dim2)
   c1 = my_new_rank - c0*dim1    #              c1 = (my_new_rank - c0*(dim1*dim2) / dim2
                                 #              c2 = my_new_rank - c0*(dim1*dim2) - c1*dim2 
   # coordinates through a cartesian Virtual toplogy based on new_comm
   dims[0] = dim0; dims[1] = dim1;  periods[0]=True; periods[1]=True
   comm_cart = new_comm.Create_cart(dims=dims, periods=periods, reorder=False)  
   coords = comm_cart.Get_coords(my_new_rank)

   # comparison of the results
   if (c0 != coords[0]):
      print("NEWrank={:3d}, ERROR in coords[0]: {:2d} != {:2d}".format(my_new_rank, c0, coords[0]))
   if (c1 != coords[1]):
      print("NEWrank={:3d}, ERROR in coords[1]: {:2d} != {:2d}".format(my_new_rank, c1, coords[1]))

   # Ring in direction 0, i.e., with different c0 (and same other coord(s))
   # ----------------------------------------------------------------------

   # Source code without Cartesian Virtual Toplogy
   left__coord0 = (c0-1+dim0) % dim0;   left__rank = left__coord0*dim1+c1;
   right_coord0 = (c0+1)      % dim0;   right_rank = right_coord0*dim1+c1;   
   # coordinates through a cartesian Virtual toplogy based on new_comm
   # right=..., left=... should be substituted by one call to MPI_Cart_shift():
   (left, right) = comm_cart.Shift(direction=0, disp=1) 
   # comparison of the results
   if (left__rank != left ):
      print("DIR=0, NEWrank={:3d}, ERROR in left:  {:2d} != {:2d}".format(my_new_rank, left__rank, left))
   if (right_rank != right):
      print("DIR=0, NEWrank={:3d}, ERROR in right: {:2d} != {:2d}".format(my_new_rank, right_rank, right))

   sum = 0
   snd_buf = np.array(my_new_rank, dtype=np.intc) 

   for i in range(dim0):
      # without Cartesian Virtuzal Topology, comm_cart must be substituted by new-comm
      request = comm_cart.Issend((snd_buf, 1, MPI.INT), right, to_right)
      comm_cart.Recv((rcv_buf, 1, MPI.INT), left, to_right, status)
      request.Wait(status)
      np.copyto(snd_buf, rcv_buf)
      sum += rcv_buf
   print("DIR=0, RANK world: {:3d} new: {:3d} -- coords[0]: {:2d} [1]: {:2d} -- left= {:3d} right= {:3d} -- sum= {:4d}".format( 
                   my_world_rank, my_new_rank,coords[0],    coords[1],   left,     right,        sum))

   # Ring in direction 1, i.e., with different c1 (and same other coord(s))
   # ----------------------------------------------------------------------

   # without Cartesian Virtuzal Topology, comm_cart must be substituted by new-comm
   left__coord1 = (c1-1+dim1) % dim1;   left__rank = c0*dim1 + left__coord1
   right_coord1 = (c1+1)      % dim1;   right_rank = c0*dim1 + right_coord1
   # coordinates through a cartesian Virtual toplogy based on new_comm
   # right=..., left=... should be substituted by one call to MPI_Cart_shift():
   (left, right) = comm_cart.Shift(direction=1, disp=1)
   # comparison of the results
   if (left__rank != left ): 
      print("DIR=1, NEWrank={:3d}, ERROR in left:  {:2d} != {:2d}".format(my_new_rank, left__rank, left))
   if (right_rank != right): 
      print("DIR=1, NEWrank={:3d}, ERROR in right: {:2d} != {:2d}".format(my_new_rank, right_rank, right))

   sum = 0
   snd_buf = np.array(my_new_rank, dtype=np.intc)

   for i in range(dim1):
      # without Cartesian Virtuzal Topology, comm_cart must be substituted by new-comm
      request = comm_cart.Issend((snd_buf, 1, MPI.INT), right, to_right)
      comm_cart.Recv((rcv_buf, 1, MPI.INT), left, to_right, status)
      request.Wait(status)
      np.copyto(snd_buf, rcv_buf)
      sum += rcv_buf
   print("DIR=1, RANK world: {:3d} new: {:3d} -- coords[0]: {:2d} [1]: {:2d} -- left= {:3d} right= {:3d} -- sum= {:4d}".format( 
                   my_world_rank, my_new_rank,coords[0],    coords[1],   left,     right,        sum))

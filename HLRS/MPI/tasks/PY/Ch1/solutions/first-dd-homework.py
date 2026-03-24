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
#  Purpose: A first MPI example calculating the subdomain size  # 
#                                                               # 
#  Contents: Python code, object send version (comm.send)       # 
#                                                               # 
#################################################################

from mpi4py import MPI

idim: int =3 # = number of processes in this dimension
icoord: int  # = number of a process in this dimension: between 0 and idim-1
istart: int =0; imax: int =80 # = start and last index of the global data mesh
                              #   in one dimension, including the boundary condition.
isize: int     # = length of the global data mesh in this dimension, i.e.,
               # = imax-istart+1
b1: int=1  # Width of the boundary condition = width of the halos
           # in this dimension.
iss: int; ie: int # start and end index of the subarray (of the domain decomposition)
                  # in this dimension in process icoord 
iouter: int  # = ie-iss+1 = size of the local subdomain data mesh in that dimension
iinner: int  # = iouter-2*b1 = number of unknowns in this dimension of process icoord
iinner0: int # = smallest number of unknowns in a process in this dimension
in1: int     # = number of processes with inner=inner0+1

numprocs: int; my_rank: int # additional variables for parallelization

# Setting: This program should run only with one MPI process and calculates the
#          subarray length  iouter  and its indices from  is  to  ie  (as in the global array)
#          for each of the  idim  processes in one of dimension,
#          i.e., for process coordinate  icoord  between  0  and  idim-1 
# Given:   idim, istart, imax, (and therefore isize), b1
# Goal:    to calculate iss, ie (and iouter and iinner) for each icoord between 0 and idim-1

comm_world = MPI.COMM_WORLD
my_rank = comm_world.Get_rank()
num_procs = comm_world.Get_size()

if (my_rank==0):
    # run this test only with one MPI process

    print(f"\nType in imax (=last index of the global data mesh) and")
    print(  f"idim (=number of processes in a dimension) and")
    print(  f"b1 (=width of boundary = halo width), e.g. {imax} {idim} {b1}")

    split_str=input()
    if not split_str:
        print("--- no input, defaults are used ---")
    else:
        (imax, idim, b1) = tuple(int(element) for element in split_str.split(None,2))
        # Closer to scanf with additional module is:
        # from parse import parse
        # (imax, idim, b1) = parse("{:d} {:d} {:d}", split_str)
    print(f"\nindices of the global data mesh are between {istart} and {imax}")
    print(  f"number of processes in this dimension = {idim}")
    print(  f"boundary width = halo width is {b1}")
    print(  f"indices of the unknowns are between {istart+b1} and {imax-b1}")

    for icoord in range(0, idim):
        # emulating all processes in one dimension 


# Calculating the own subdomain in each process
# ---------------------------------------------
#
# whole indices      |------------- isize = imax-istart+1 ------------|
#    start/end index  ^-istart                                  imax-^
# 
# 1. interval        |--- iouter1---|   
#                    |--|--------|--|
#                     b1  iinner1 b1 
#    start/end index  ^-iss     ie-^ 
# 2. interval                 |--|--------|--|   
# 3. interval                          |--|--------|--| 
# 4. interval                                   |--|-------|--| 
#                                                   iinner0 = iinner1 - 1
# 5. interval = idim's interval                         |--|-------|--|
#
# In each iteration on each interval, the inner area is computed
# by using the values of the last iteration in the whole outer area. 
#
# icoord = number of the interval - 1
# 
# To fit exactly into isize, we use in1 intervals of with iinner1 = iinner0 + 1
# and (idim-in1) intervals of with iinner0 
#
#         Originally:     And as result of the domain decomposition into idim subdomains:
# isize = imax-istart+1 = 2*b1 + in1*iinner1 + (idim-in1)*inner0
#
# computing iss:ie, ks:ke
#   - input:            istart, imax, b1, idim, icoord (and k...)
#   - to be calculated: iss, ie, iinner, iouter
#   - helper variables: iinner0, in1, isize

        isize = imax - istart + 1; # total number of elements, including the "2*b1" boundary elements
        # isize - 2*b1 = total number of unknowns
        iinner0 = (isize - 2*b1)  // idim # smaller inner size through divide with rounding off
        in1 = isize - 2*b1 - idim * iinner0 # number of processes that must have "inner0+1" unknowns
        if (icoord < in1):  # the first in1 processes will have "iinner0+1" unknowns
            iinner = iinner0 + 1
            iss = (istart+b1) + icoord * iinner - b1 # note that "is" reflects the position of the 
                                                     # first halo or boundary element of the subdomain
        else:               # and all other processes will have iinner0 unknowns
            iinner = iinner0
            iss = istart + in1 * (iinner0+1) + (icoord-in1) * iinner
        
        iouter = iinner + 2*b1;
        ie = iss + iouter - 1;

        if (icoord==0):
            print( "\nPlease control whether isize and the sum are identical:")
            print("  isize={:3d} idim={:3d} || in1*(iinner0+1)={:3d} *{:3d} + (idim-in1)*iinner0={:3d} *{:3d} + 2*b1=2*{:1d} || sum = {:3d}".format(
                            isize,    idim,                   in1, iinner0+1,                   idim-in1, iinner0,   b1,
                                                                                                                                       in1*(iinner0+1) + (idim-in1)*iinner0 + 2*b1))
            print("\nPlease control whether the indices of unkowns are between {:d}..{:d}, complete, and non-overlapping:".format(
                                                                          istart+b1, imax-b1))
    
        if (iinner>0):
            print("  icoord={:d}, iouter={:2d}, iinner={:2d}, subarray indices= {:2d}..{:2d}, indices of the unknowns= {:2d}..{:2d}".format(
                            icoord,       iouter,      iinner,                    iss, ie,                            iss+b1, ie-b1))
        else:
            print("  icoord={:d}, iouter={:2d}, iinner={:2d}, no subarray".format(
                      icoord,    iouter,     iinner ))

    # end for icoord...
# end if (myrank==0)

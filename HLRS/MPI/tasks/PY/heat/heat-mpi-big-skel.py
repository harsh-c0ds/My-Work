#!/usr/bin/env python3

# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
from mpi4py import MPI
import numpy as np
import os

def idx(i,k):
    return (((i)-iss)  *kouter + (k)-ks)
def idxn(i,k):
    return (((i)-iss-1)*kinner + (k)-ks-1)

prt=True; prt_halo=True; # True or False for printing / not printing the result / plus halos
imax=15; kmax=12; istart=0; kstart=0; b1=1; it=0; itmax=20000;
eps=1.e-08;

# Preparation for parallelization with domain decomposition:

# int is=istart, ie=imax, ks=kstart, ke=kmax; # now defined and calculated below

# The algortithm below is already formulated with lower and upper 
# index values (is:ie), (ks:ke) instead of (istart:imax), (kstart:kmax).
# With the MPI domain decomposition, (is:ie) and (ks:ke) will define the index rang 
# of the subdomain in a given MPI process.

# additional variables for parallelization

dims = [0]*2; coords = [0]*2; # helper variables only for MPI_CART_CREATE and ..._COORDS
period = [False]*2;           #                  only for MPI_CART_CREATE and ..._COORDS
stride=10; # for reduced calculation of the abort criterion
req = [MPI.Request()]*4;
statuses = [MPI.Status()]*4;
gsizes = [0]*2; lsizes = [0]*2; starts=[0]*2; # only for MPI_Type_create_subarray
dphimaxpartial = np.empty((), dtype=np.double)




# [empty space because in C and Fortran, additional declarations are needed]



#  naming: originally: i=istart..imax,      now: i=is..ie,  icoord=0..idim-1
#          originally: k=kstart..kmax,      now: k=ks..ke,  kcoord=0..kdim-1
#                      with istart=kstart=0      s=start index,  e=end index

comm_world = MPI.COMM_WORLD
my_rang = comm_world.Get_rank();
numprocs = comm_world.Get_size();

# the 2-dimensional domain decomposition:

dims[0] = 0; dims[1] = 0;
period[0] = False; period[1] = False;
dims = MPI.Compute_dims(numprocs, dims);
idim = dims[0];
kdim = dims[1];
comm = comm_world.Create_cart(dims,period,True)
my_rank = comm.Get_rank()
coords = comm.Get_coords(my_rank)
icoord = coords[0]
kcoord = coords[1]
(left,right)  = comm.Shift(0, 1)
# the ranks (left,right) represent the coordinates ((icoord-1,kcoord), (icoord+1,kcoord)) 
(lower,upper) = comm.Shift(1, 1)
# the ranks (lower,upper) represent the coordinates ((icoord,kcoord-1),(icoord,kcoord+1))

# Exercise step 1: calculating the own subdomain in each process
# --------------------------------------------------------------
#
# whole y indecees   |------------- isize = imax-istart+1 ------------|
#    start/end index  ^-istart                                  imax-^
# 
# 1. interval        |--- iouter1---|   
#                    |--|--------|--|
#                     b1  iinner1 b1 
#    start/end index  ^-is      ie-^ 
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
# computing is:ie, ks:ke
#   - input:            istart, imax, b1, idim, icoord (and k...)
#   - to be calculated: is, ie, iinner, iouter
#   - helper variables: iinner0, in1, isize

isize = imax - istart + 1; # total number of elements, including the "2*b1" boundary elements
# isize - 2*b1 = total number of unknowns
# iinner0 = ________________________ # smaller inner size through divide with rounding off
# in1 = ______________________________ # number of processes that must have "inner0+1" unknowns
if (icoord < in1): # the first in1 processes will have "iinner0+1" unknowns
    # iinner = ____________
    # iss = ___________________________________ # note that "is" reflects the position of the 
    pass                                     # first halo or boundary element of the subdomain
else:              # and all other processes will have iinner0 unknowns
    # iinner = ________
    # iss = ___________________________________________________
    pass
# iouter = ______________
# ie = _________________

# same for x coordinate: 
ksize = kmax - kstart + 1; # total number of elements, including the "2*b1" boundary elements
# ksize - 2*b1 = total number of unknowns
# kinner0 = ________________________ # smaller inner size through divide with rounding off
# kn1 = ______________________________ # number of processes that must have "knner0+1" unknowns
if (kcoord < kn1) :  # the first kn1 processes will have "kinner0+1" unknowns
    # kinner = ____________
    # ks = ___________________________________ # note that "ks" reflects the position of the 
    pass                                     # first halo or boundary element of the subdomain
else:              # and all other will have kinner0 unknowns
    # kinner = ________
    # ks = ___________________________________________________
    pass
# kouter = ______________
# ke = ________________

if (my_rank == 0) :
    print("\n isize={:3d} idim={:3d} || in1*(iinner0+1)={:3d} *{:3d} + (idim-in1)*iinner0={:3d} *{:3d} + 2*b1=2*{:1d} || sum = {:3d}".format(
                          isize,      idim,         in1, iinner0+1,                 idim-in1, iinner0,                 b1,
                                                                                                  in1*(iinner0+1) + (idim-in1)*iinner0 + 2*b1))
    print("\n ksize={:3d} kdim={:3d} || kn1*(kinner0+1)={:3d} *{:3d} + (kdim-kn1)*kinner0={:3d} *{:3d} + 2*b1=2*{:1d} || sum = {:3d}".format(
                              ksize,      kdim,         kn1, kinner0+1,                 kdim-kn1, kinner0,                 b1,
                                                                                                                        kn1*(kinner0+1) + (kdim-kn1)*kinner0 + 2*b1))




# It must be guaranteed that the array 'phin' is not an empty array,
# i.e., that the iinner in all processes is at least 1,
# i.e., the total number of unknowns in direction "i" is at least 1*idim, and same for k, 
# because otherwise, the halo communication would not work.
# Note that all processes have to make the same decision. Therefore they should
# use "isize,ksize" and "idim,kdim" and cannot use only their own "iinner,kinner".

if (((isize - 2*b1) < idim) or ((ksize - 2*b1) < kdim)):
    if(my_rank == 0):
        printf("phin is in some processes an empty array because isize-2*b1={:3d} < idim={:3d} or ksize-2*b1={:3d} < kdim={:3d".format(
                                                                 isize-2*b1,        idim,         ksize-2*b1,        kdim))
    sys.exit(0)




phi  = np.empty(iouter*kouter, dtype=np.double) # index range: (is:ie,ks:ke)
phin = np.empty(iinner*kinner, dtype=np.double) # index range: (is+b1:ie-b1,ks+b1:ke-b1)

# create and commit derived datatypes vertical_border and horizontal_border through MPI_Type_vector()
vertical_border = MPI.DOUBLE.Create_vector(b1,kinner,kouter);
vertical_border.Commit()
horizontal_border = MPI.DOUBLE.Create_vector(iinner,b1,kouter);
horizontal_border.Commit()

# Exercise step 4: Advanced exercise:  Same as with MPI_Type_vector(), but with MPI_Type_create_subarray()
# ----------------------------------   (advantage: would work also with 3,4, ... dimensions)
#
# gsizes[0]=______; gsizes[1]=____________
# lsizes[0]=______; lsizes[1]=______;  starts[0]=0; starts[1]=0;
# vertical_border = __________.Create_subarray(gsizes, lsizes, starts, order=___________)
# vertical_border.Commit()
# gsizes[0]=______; gsizes[1]=______; 
# lsizes[0]=______; lsizes[1]=______;  starts[0]=0; starts[1]=0;
# horizontal_border = __________.Create_subarray(gsizes, lsizes, starts, order=___________)
# horizontal_border.Commit()


# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
dx=1.e0/(kmax-kstart);
dy=1.e0/(imax-istart);
dx2=dx*dx;
dy2=dy*dy;
dx2i=np.double(1.e0/dx2);
dy2i=np.double(1.e0/dy2);
dt=np.double(min(dx2,dy2)/4.e0);
# start values 0.d0 
for i in range(max(1,iss), min(ie,imax-1)+1) : # do not overwrite the boundary condition
    for k in range(ks,min(ke,kmax-1)+1):
        phi[idx(i,k)]=0.e0;
    # end for
# end for
# start values 1.d0
if (ke == kmax): 
    for i in range(iss,ie+1):
        phi[idx(i,kmax)]=1.e0;
    # end for

# start values dx
if (iss == istart): 
# phi[idx(0,0)]=0.d0
# for(k=1; k<=kmax-1 ; k++): 
#     phi[idx(0,k)]=phi[idx(0,k-1)]+dx 
#     phi[idx(imax,k)]=phi[idx(imax,k-1)]+dx 
# # end for 
# ... substitute algorithmus by a code, 
#     that can be computed locally: 
    for k in range(ks,min(ke,kmax-1)+1): 
        phi[idx(istart,k)] = (k-kstart)*dx;
    # end for

if (ie == imax): 
    for k in range(ks, min(ke,kmax-1)+1):
        phi[idx(imax,k)] = (k-kstart)*dx;
    # end for


# print details
if (my_rank == 0): 
    print("\nHeat Conduction 2d")
    print(f"\ndx ={dx:12.4e}, dy ={dy:12.4e}, dt={dt:12.4e}, eps={eps:12.4e}")

start_time = MPI.Wtime();
comm_time = 0.0;
criterion_time = 0.0;

# iteration
for it in range(1,itmax+1):
    dphimax: np.ndarray=np.zeros((),dtype=np.double)
    for i in range(iss+b1,ie-b1+1):
        for k in range(ks+b1,ke-b1+1):
            dphi=(   (phi[idx(i+1,k)]+phi[idx(i-1,k)]-2.*phi[idx(i,k)])*dy2i
                    +(phi[idx(i,k+1)]+phi[idx(i,k-1)]-2.*phi[idx(i,k)])*dx2i);
            dphi=dphi*dt;
            np.copyto(dst=dphimax,src=np.max(dphi,initial=dphimax))
            phin[idxn(i,k)]=phi[idx(i,k)]+dphi;
        # end for
    # end for
# save values
    for i in range(iss+b1,ie-b1+1):
        for k in range(ks+b1,ke-b1+1):
            phi[idx(i,k)]=phin[idxn(i,k)];
        # end for
    # end for

# Exercise step 2: common abort criterion for all processes
# ---------------------------------------------------------
#
# for optimization: allreduce only each stride's loop:
    criterion_time = criterion_time - MPI.Wtime();
    if ((it % stride) == 0):
        if (numprocs > 1): 
            # the following code is necessary, because each process only calculates a local (and not global) dphimax
            # ... 
            # ... 
            pass
        if(dphimax < eps):
            criterion_time = criterion_time + MPI.Wtime();
            break; # Finish the timestep loop "do it=…"


    criterion_time = criterion_time + MPI.Wtime();

# Exercise step 3: the halo communication
# ---------------------------------------
#
    comm_time = comm_time - MPI.Wtime();
# send and receive to/from upper/lower
    if (kdim > 1) : # otherwise in all processes both, lower and upper are MPI_PROC_NULL
#         ______ = comm....
#         ______ = comm.... please use nonblocking recv and nonblocking send
#         ______ = comm....
#         ______ = comm....
#         MPI._______
        # or alternatively with sendrecv:
        # 
        # not part of the regular exercise
        #
        #
        pass

# send and receive to/from left/right
    if (idim > 1) : # otherwise in all processes both, left and right are MPI_PROC_NULL
#         ______ = comm....
#         ______ = comm.... please use nonblocking recv and nonblocking send
#         ______ = comm....
#         ______ = comm....
#         MPI._______
        # or alternatively with sendrecv:
        # 
        # not part of the regular exercise
        #
        #
        pass
    comm_time = comm_time + MPI.Wtime();

# end for

end_time = MPI.Wtime();

if (prt) :
    for ic in range(0,idim):
        for kc in range(0,kdim):
            if ((ic == icoord) and (kc == kcoord)) :
                i_first = iss
                i_last  = ie
                k_first = ks
                k_last  = ke
                if not prt_halo:
                    if (ic > 0) :
                        i_first = iss + b1    # do not print halo at the beginning
                    if (ic < idim-1) :
                        i_last = ie - b1      # do not print halo at the end
                    if (kc > 0) :
                        k_first = ks + b1     # do not print halo at the beginning
                    if (kc < kdim-1) :
                        k_last = ke - b1      # do not print halo at the end

                if (kc == 0):
                    print(f"\nprinting the {ic:3d}th horizontal block")
                    print("               i=",end='')

                    for i in range(i_first,i_last+1):
                        print(f" {i:4d}",end='')
                    print("")
                else:
                    if (prt_halo):
                        print(""); # additional empty line between the processes

                for k in range(k_first,k_last+1):
                    print(f"ic={ic:2d} kc={kc:2d} k={k:3d}",end='')
                    for i in range(i_first,i_last+1):
                        print(f" {phi[idx(i,k)]:4.2f}",end='')
                    print("")
                # end for
            comm.Barrier   # to separate the printing of each block by different processes.
                           # Caution: This works in most cases, but does not guarantee that the
                           #          output lines on the common stdout are in the expected sequence.
        # end for
    # end for
# end if prt

if (my_rank == 0):
    print("\n!numprocs=idim    iter-   wall clock time  communication part  abort criterion")
    print(  "!          x kdim ations     [seconds]     method [seconds]    meth. stride [seconds]")
    print(  "!{:6d} ={:3d} x{:3d} {:6d} {:12.4g}      {:2d} {:12.4g}    {:2d} {:6d} {:12.4g}".format(
          numprocs, idim, kdim, it,  end_time - start_time,
                                                          1, comm_time, 1, stride, criterion_time))

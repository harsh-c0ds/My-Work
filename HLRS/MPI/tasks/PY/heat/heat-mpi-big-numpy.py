#!/usr/bin/env python3

# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
from mpi4py import MPI
import numpy as np
import os






i:int; k:int; it:int=0;
prt:bool=True; prt_halo:bool=True; # True or False for printing / not printing the result / plus halos
imax:int=80; kmax:int=80; istart:int=0; kstart:int=0; b1:int=1; itmax:int=20000;
eps:np.double=np.double(1.e-08);
#dx,dy,dx2,dy2,dx2i,dy2i,dt,dphi;

# Preparation for parallelization with domain decomposition:

# is:int=istart; ie:int=imax; ks:int=kstart; ke:int=kmax; # now defined and calculated below

# The algortithm below is already formulated with lower and upper 
# index values (is:ie), (ks:ke) instead of (istart:imax), (kstart:kmax).
# With the MPI domain decomposition, (is:ie) and (ks:ke) will define the index rang 
# of the subdomain in a given MPI process.

# additional variables for parallelization
numprocs:int; my_rank:int; right:int; left:int; upper:int; lower:int;
comm:MPI.Comm;  # Cartesian communicator
dims:list[int]=[0]*2; coords:list[int]=[0]*2; # helper variables only for MPI_CART_CREATE and ..._COORDS
period:list[bool]=[False]*2;                  # only for MPI_CART_CREATE and ..._COORDS
idim:int; kdim:int; icoord:int; kcoord:int;
isize:int; iinner0:int; in1:int;  ksize:int; kinner0:int; kn1:int; # only for calculation of iinner, ...
iinner:int; iouter:int; iss:int; ie:int; kinner:int; kouter:int; ks:int; ke:int;
ic:int; i_first:int; i_last:int;  kc:int; k_first:int; k_last:int; # only for printing
stride:int=10; # for reduced calculation of the abort criterion
req:list[MPI.Request]     = [MPI.Request()]*4;
statuses:list[MPI.Status] = [MPI.Status()]*4;
gsizes:list[int]=[0]*2; lsizes:list[int]=[0]*2; starts:list[int]=[0]*2; # only for MPI_Type_create_subarray
horizontal_border: MPI.Datatype; vertical_border: MPI.Datatype # datatype
dphimaxpartial: np.ndarray = np.empty((),dtype=np.double) # for local/global calculation of the abort criterion
start_time: float; end_time: float; comm_time: float; criterion_time: float
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
iinner0 = (isize - 2*b1)  // idim; # smaller inner size through divide with rounding off
in1 = isize - 2*b1 - idim * iinner0; # number of processes that must have "inner0+1" unknowns
if (icoord < in1): # the first in1 processes will have "iinner0+1" unknowns
    iinner = iinner0 + 1;
    iss = (istart+b1) + icoord * iinner - b1; # note that "is" reflects the position of the 
                                             # first halo or boundary element of the subdomain
else:              # and all other processes will have iinner0 unknowns
    iinner = iinner0;
    iss = istart + in1 * (iinner0+1) + (icoord-in1) * iinner;

iouter = iinner + 2*b1;
ie = iss + iouter - 1;

# same for x coordinate: 
ksize = kmax - kstart + 1; # total number of elements, including the "2*b1" boundary elements
# ksize - 2*b1 = total number of unknowns
kinner0 = (ksize - 2*b1)  // kdim; # smaller inner size through divide with rounding off
kn1 = ksize - 2*b1 - kdim * kinner0; # number of processes that must have "knner0+1" unknowns
if (kcoord < kn1) :  # the first kn1 processes will have "kinner0+1" unknowns
    kinner = kinner0 + 1;
    ks = (kstart+b1) + kcoord * kinner - b1; # note that "ks" reflects the position of the 
                                             # first halo or boundary element of the subdomain
else:              # and all other will have kinner0 unknowns
    kinner = kinner0;
    ks = kstart + kn1 * (kinner0+1) + (kcoord-kn1) * kinner;

kouter = kinner + 2*b1;
ke = ks + kouter - 1;

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




phi  = np.empty((iouter,kouter), dtype=np.double) # index range: (is:ie,ks:ke)


# create and commit derived datatypes vertical_border and horizontal_border through MPI_Type_vector()
# vertical_border = MPI.DOUBLE.Create_vector(b1,kinner,kouter);
# vertical_border.Commit()
# horizontal_border = MPI.DOUBLE.Create_vector(iinner,b1,kouter);
# horizontal_border.Commit()

# Exercise step 4: Advanced exercise:  Same as with MPI_Type_vector(), but with MPI_Type_create_subarray()
# ----------------------------------   (advantage: would also work with 3,4, ... dimensions)
#
gsizes[0]=iouter; gsizes[1]=kouter; 
lsizes[0]=b1;     lsizes[1]=kinner;  starts[0]=0; starts[1]=0;
vertical_border = MPI.DOUBLE.Create_subarray(gsizes, lsizes, starts, order=MPI.ORDER_C)
vertical_border.Commit()
gsizes[0]=iouter; gsizes[1]=kouter; 
lsizes[0]=iinner; lsizes[1]=b1;      starts[0]=0; starts[1]=0;
horizontal_border = MPI.DOUBLE.Create_subarray(gsizes, lsizes, starts, order=MPI.ORDER_C)
horizontal_border.Commit()


# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
dx:  float=1.e0/(kmax-kstart);
dy:  float=1.e0/(imax-istart);
dx2: float=dx*dx;
dy2: float=dy*dy;
dx2i:np.double=np.double(1.e0/dx2);
dy2i:np.double=np.double(1.e0/dy2);
dt:  np.double=np.double(min(dx2,dy2)/4.e0);
# start values 0.d0 
phi[max(1,iss)-iss:min(ie,imax-1)+1-iss,0:min(ke,kmax-1)+1-ks]=float(0.0); # do not overwrite the boundary condition




# start values 1.d0
if (ke == kmax): 
    phi[0:ie+1-iss,kmax-ks]=float(1.0);


# start values dx
if (iss == istart): 
#  phi[0,0]=0.e0
#  phi[1:kmax,0]    = np.arange(1:kmax, dtype=np.intc)*dx
#  phi[1:kmax,imax] = np.arange(1:kmax, dtype=np.intc)*dx
#  ... substitute algorithmus by a code, 
#      that can be computed locally:
    phi[istart-iss,0:min(ke,kmax-1)+1-ks] = np.arange(ks-kstart,min(ke,kmax-1)+1-kstart, dtype=np.intc)*dx;






if (ie == imax): 
    phi[imax-iss,0:min(ke,kmax-1)+1-ks] = np.arange(ks-kstart,min(ke,kmax-1)+1-kstart, dtype=np.intc)*dx;




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
    dphi=(  (    phi[b1+1:ie-b1+1-iss+1,b1:ke-b1+1-ks]
                +phi[b1-1:ie-b1+1-iss-1,b1:ke-b1+1-ks]
             -2.*phi[b1  :ie-b1+1-iss  ,b1:ke-b1+1-ks])*dy2i
           +(    phi[b1:ie-b1+1-iss,b1+1:ke-b1+1-ks+1]
                +phi[b1:ie-b1+1-iss,b1-1:ke-b1+1-ks-1]
             -2.*phi[b1:ie-b1+1-iss,b1  :ke-b1+1-ks  ])*dx2i
         )
    dphi=dphi*dt;
    np.copyto(dst=dphimax,src=np.max(dphi,initial=0.0))
    # save values
    phi[b1:ie-b1+1-iss,b1:ke-b1+1-ks]=phi[b1:ie-b1+1-iss,b1:ke-b1+1-ks]+dphi;





# Exercise step 2: common abort criterion for all processes
# ---------------------------------------------------------
#
# for optimization: allreduce only each stride's loop:
    criterion_time = criterion_time - MPI.Wtime();
    if ((it % stride) == 0):
        if (numprocs > 1): 
            # the following code is necessary, because each process only calculates a local (and not global) dphimax
            np.copyto(dst=dphimaxpartial, src=dphimax)
            comm.Allreduce(sendbuf=dphimaxpartial, recvbuf=(dphimax, 1, MPI.DOUBLE), op=MPI.MAX)
        
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
        req[0] = comm.Irecv((phi[b1,           0:],1,horizontal_border), source=lower,tag=1)
        req[1] = comm.Irecv((phi[b1,  ke+1-b1-ks:],1,horizontal_border), source=upper,tag=2)
        req[2] = comm.Isend((phi[b1,ke+1-2*b1-ks:],1,horizontal_border), dest=upper,  tag=1)
        req[3] = comm.Isend((phi[b1,          b1:],1,horizontal_border), dest=lower,  tag=2)
        MPI.Request.Waitall(req, statuses)
        # or alternatively with sendrecv:
        # comm.Sendrecv((phi[b1,ke+1-2*b1-ks:],1,horizontal_border), dest=upper,  sendtag=1,)
        #               (phi[b1,           0:],1,horizontal_border), source=lower,recvtag=1, MPI.STATUS_IGNORE)
        # comm.Sendrecv((phi[b1,          b1:],1,horizontal_border), dest=lower,  sendtag=2,
        #               (phi[b1,  ke+1-b1-ks:],1,horizontal_border), source=upper,recvtag=2, MPI.STATUS_IGNORE)


# send and receive to/from left/right
    if (idim > 1) : # otherwise in all processes both, left and right are MPI_PROC_NULL
        req[0] = comm.Irecv((phi[0,            b1:],1,vertical_border), source=left, tag=3)
        req[1] = comm.Irecv((phi[ie+1-b1-iss,  b1:],1,vertical_border), source=right,tag=4)
        req[2] = comm.Isend((phi[ie+1-2*b1-iss,b1:],1,vertical_border), dest=right,  tag=3)
        req[3] = comm.Isend((phi[b1,           b1:],1,vertical_border), dest=left,   tag=4)
        MPI.Request.Waitall(req, statuses);
        # or alternatively with sendrecv:
        # comm.Sendrecv((phi[ie+1-2*b1-iss,b1:],1,vertical_border), dest=right,  tag=3,
        #               (phi[0,            b1:],1,vertical_border), source=left, tag=3, MPI.STATUS_IGNORE)
        # comm.Sendrecv((phi[b1,           b1:],1,vertical_border), dest=left,   tag=4,
        #               (phi[ie+1-b1-iss,  b1:],1,vertical_border), source=right,tag=4, MPI.STATUS_IGNORE)

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
                        print(f" {phi[i-iss,k-ks]:4.2f}",end='')
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

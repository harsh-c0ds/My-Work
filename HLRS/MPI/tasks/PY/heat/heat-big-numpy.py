#!/usr/bin/env python3

import numpy as np
import time

# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
i: int; k: int; it: int=0
prt: bool=True; # True or False for printing / not printing the result
imax: int=80; kmax: int=80; istart: int=0; kstart: int=0; b1: int=1; itmax: int=20000
eps: np.double= np.double(1.e-08)

# Preparation for parallelization with domain decomposition:
iss: int=istart; ie: int=imax; ks: int=kstart; ke: int=kmax # now defined and calculated below
iouter: int=ie-iss+1; kouter: int=ke-ks+1; iinner: int=iouter-2*b1; kinner: int=kouter-2*b1

# The algortithm below is already formulated with lower and upper 
# index values (iss:ie), (ks:ke) instead of (istart:imax), (kstart:kmax).
# With the MPI domain decomposition, (iss:ie) and (ks:ke) will define the index rang 
# of the subdomain in a given MPI process.

phi  = np.empty((iouter,kouter), dtype=np.double) # index range: (iss:ie,ks:ke)

#         
# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
dx:  float=1.e0/(kmax-kstart)
dy:  float=1.e0/(imax-istart)
dx2: float=dx*dx
dy2: float=dy*dy
dx2i: np.double=np.double(1.e0/dx2)
dy2i: np.double=np.double(1.e0/dy2)
dt:   np.double=np.double(min(dx2,dy2)/4.e0)
# start values 0.d0 
phi[max(1,iss):min(ie,imax-1)+1,ks:min(ke,kmax-1)+1] = np.double(0.0) # do not overwrite the boundary condition

# start values 1.d0
if (ke == kmax):
    phi[iss:ie+1,kmax] = np.double(1.0)

# start values dx
if (iss == istart):
#  phi[0,0]=0.e0
#  phi[0,1:kmax]    = np.arange(1:kmax, dtype=np.intc)*dx
#  phi[imax,1:kmax] = np.arange(1:kmax, dtype=np.intc)*dx
#  ... substitute algorithmus by a code, 
#      that can be computed locally: 
    phi[istart,ks:min(ke,kmax-1)+1] = np.arange(ks-kstart,min(ke,kmax-1)+1, dtype=np.intc)*dx

if (ie == imax):
    phi[imax,ks:min(ke,kmax-1)+1] = np.arange(ks-kstart,min(ke,kmax-1)+1, dtype=np.intc)*dx

# print details
print("\nHeat Conduction 2d")
print(f"\ndx ={dx:12.4e}, dy ={dy:12.4e}, dt={dt:12.4e}, eps={eps:12.4e}") 

t=time.perf_counter_ns()

# iteration
for it in range(1, itmax+1):
    #dphimax=0.;
    # k: b1, ke-b1-ks; i: b1, ie-b1-iss;
    dphi=((      phi[b1+1:ie-b1+1-iss+1,b1:ke-b1+1-ks]
               + phi[b1-1:ie-b1+1-iss-1,b1:ke-b1+1-ks]
            -2.0*phi[b1  :ie-b1+1-iss  ,b1:ke-b1+1-ks])*dy2i 
           +(    phi[b1:ie-b1+1-iss,b1+1:ke-b1-ks+2] 
                +phi[b1:ie-b1+1-iss,b1-1:ke-b1-ks  ]
            -2.0*phi[b1:ie-b1+1-iss,b1  :ke-b1-ks+1])*dx2i
         )
    dphi = dphi*dt
    # save values
    phi[iss+b1:ie-b1+1,ks+b1:ke-b1+1] = phi[iss+b1:ie-b1+1,ks+b1:ke-b1+1] + dphi
    if np.max(dphi,initial=0.0) < eps:
        break # Finish the timestep loop "do it=…"

if prt == True:
    print("   i=", end='');
    for i in range(iss,ie+1):
        print(f" {i:4d}",end='')
    print("")
    for k in range(ks,ke+1):
        print(f"k={k:3d}", end='')
        for i in range(iss,ie+1):
            print(f" {phi[i,k]:4.2f}", end='')
        print("")
    # for
# if prt

print(f"\n{it:d} iterations")
print(f"\nCPU time = {(time.perf_counter_ns()-t)*1e-9:#12.4g} sec")

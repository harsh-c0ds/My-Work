#!/usr/bin/env python3

import numpy as np
import time

def idx(i,k):
    return (((i)-iss)  *kouter + (k)-ks)

def idxn(i,k):
    return (((i)-iss-1)*kinner + (k)-ks-1)

# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
prt=True; # True or False for printing / not printing the result
imax=40; kmax=40; istart=0; kstart=0; b1=1; it=0; itmax=20000;
eps=1.e-08;


# Preparation for parallelization with domain decomposition:
iss=istart; ie=imax; ks=kstart; ke=kmax; # now defined and calculated below
iouter=ie-iss+1; kouter=ke-ks+1; iinner=iouter-2*b1; kinner=kouter-2*b1;

# The algortithm below is already formulated with lower and upper 
# index values (is:ie), (ks:ke) instead of (istart:imax), (kstart:kmax).
# With the MPI domain decomposition, (is:ie) and (ks:ke) will define the index rang 
# of the subdomain in a given MPI process.

phi  = np.empty(iouter*kouter, dtype=np.double) # index range: (is:ie,ks:ke)
phin = np.empty(iinner*kinner, dtype=np.double) # index range: (is+b1:ie-b1,ks+b1:ke-b1)
#         
# naming: i = 1st array coordinate = 1st process coordinate = y
#         k = 2nd array coordinate = 2nd process coordinate = x
#                                                    Caution: y,x and not x,y
dx=1.e0/(kmax-kstart);
dy=1.e0/(imax-istart);
dx2=dx*dx;
dy2=dy*dy;
dx2i=1.e0/dx2;
dy2i=1.e0/dy2;
dt=min(dx2,dy2)/4.e0;
# start values 0.d0 
for i in range(max(1,iss),min(ie,imax-1)+1): # do not overwrite the boundary condition
    for k in range(ks,min(ke,kmax-1)+1):
      phi[idx(i,k)]=0.0;
    # end for
# end for
# start values 1.d0
if (ke == kmax): 
    for i in range(iss,ie+1):
        phi[idx(i,kmax)]=1.0;
    # end for

# start values dx
if (iss == istart): 
#   phi[idx(0,0)]=0.0
#   for k in range(1,kmax): 
#       phi[idx(0,k)]=phi[idx(0,k-1)]+dx
#       phi[idx(imax,k)]=phi[idx(imax,k-1)]+dx 
#   # end for 
#   ... substitute algorithmus by a code, 
#       that can be computed locally: 
    for k in range(ks,min(ke,kmax-1)+1):
        phi[idx(istart,k)] = (k-kstart)*dx;
    # end for

if (ie == imax): 
    for k in range(ks,min(ke,kmax-1)+1):
        phi[idx(imax,k)] = (k-kstart)*dx;
    # for


# print details
print("\nHeat Conduction 2d")
print(f"\ndx ={dx:12.4e}, dy ={dy:12.4e}, dt={dt:12.4e}, eps={eps:12.4e}")

t=time.perf_counter_ns()

# iteration
for it in range(1,itmax+1):
    dphimax=0.0;
    for i in range(iss+b1,ie-b1+1):
        for k in range(ks+b1,ke-b1+1):
            dphi=(   (phi[idx(i+1,k)]+phi[idx(i-1,k)]-2.*phi[idx(i,k)])*dy2i
                    +(phi[idx(i,k+1)]+phi[idx(i,k-1)]-2.*phi[idx(i,k)])*dx2i);
            dphi=dphi*dt;
            dphimax=max(dphimax,dphi);
            phin[idxn(i,k)]=phi[idx(i,k)]+dphi;
        # end for
    # end for
# save values
    for i in range(iss+b1,ie-b1+1):
        for k in range(ks+b1,ke-b1+1):
            phi[idx(i,k)]=phin[idxn(i,k)]
        # end for
    # end for

    if(dphimax < eps):
        break; # Finish the timestep loop "do it=…"


# end for

if prt == True:
    print("   i=", end='');
    for i in range(iss,ie+1):
        print(f" {i:4d}",end='')
    print("")
    for k in range(ks,ke+1):
        print(f"k={k:3d}", end='')
        for i in range(iss,ie+1):
            print(f" {phi[idx(i,k)]:4.2f}", end='')
        print("")
    # for
# if prt

print(f"\n{it:d} iterations")
print(f"\nCPU time = {(time.perf_counter_ns()-t)*1e-9:#12.4g} sec")






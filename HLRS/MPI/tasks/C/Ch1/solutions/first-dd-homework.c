/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the High Performance           *
 * Computing Centre Stuttgart (HLRS).                           *
 * The examples are based on the examples in the MPI course of  *
 * the Edinburgh Parallel Computing Centre (EPCC).              *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * HLRS and EPCC take no responsibility for the use of the      *
 * enclosed teaching material.                                  *
 *                                                              *
 * Authors: Rolf Rabenseifner          (HLRS)                   *
 *                                                              *
 * Contact: rabenseifner@hlrs.de                                *
 *                                                              *
 * Purpose: A second MPI example calculating the subdomain size *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int idim=3; // = number of processes in this dimension
  int icoord; // = number of a process in this dimension: between 0 and idim-1
  int istart=0, imax=80; // = start and last index of the global data mesh
                      //   in one dimension, including the boundary condition.
  int isize;     // = length of the global data mesh in this dimension, i.e.,
                 // = imax-istart+1
  int b1=1;  // Width of the boundary condition = width of the halos
             // in this dimension.
  int is, ie; // start and end index of the subarray (of the domain decomposition)
              // in this dimension in process icoord 
  int iouter;  // = ie-is+1 = size of the local subdomain data mesh in that dimension
  int iinner;  // = iouter-2*b1 = number of unknowns in this dimension of process icoord
  int iinner0; // = smallest number of unknowns in a process in this dimension
  int in1;     // = number of processes with inner=inner0+1

  int numprocs, my_rank; // additional variables for parallelization

// Setting: This program should run only with one MPI process and calculates the
//          subarray length  iouter  and its indices from  is  to  ie  (as in the global array)
//          for each of the  idim  processes in one of dimension,
//          i.e., for process coordinate  icoord  between  0  and  idim-1 
// Given:   idim, istart, imax, (and therefore isize), b1
// Goal:    to calculate is, ie (and iouter and iinner) for each icoord between 0 and idim-1

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

if(my_rank==0)
{ // run this test only with one MPI process

 printf("\nType in imax (=last index of the global data mesh) and\n");
 printf(  "idim (=number of processes in a dimension) and\n");
 printf(  "b1 (=width of boundary = halo width), e.g. %d %d %d\n",imax,idim,b1);

 int iresult= scanf("%d %d %d", &imax, &idim, &b1);
 if(iresult==EOF) printf("--- no input, defaults are used ---\n");

 printf("\nindices of the global data mesh are between %d and %d\n", istart, imax);
 printf(  "number of processes in this dimension = %d\n", idim);
 printf(  "boundary width = halo width is %d\n", b1);
 printf(  "indices of the unknowns are between %d and %d\n", istart+b1, imax-b1);

 for(icoord=0; icoord<idim; icoord++)
 { // emulating all processes in one dimension 
  
  
// Calculating the own subdomain in each process
// ---------------------------------------------
//
// whole indices      |------------- isize = imax-istart+1 ------------|
//    start/end index  ^-istart                                  imax-^
// 
// 1. interval        |--- iouter1---|   
//                    |--|--------|--|
//                     b1  iinner1 b1 
//    start/end index  ^-is      ie-^ 
// 2. interval                 |--|--------|--|   
// 3. interval                          |--|--------|--| 
// 4. interval                                   |--|-------|--| 
//                                                   iinner0 = iinner1 - 1
// 5. interval = idim's interval                         |--|-------|--|
//
// In each iteration on each interval, the inner area is computed
// by using the values of the last iteration in the whole outer area. 
//
// icoord = number of the interval - 1
// 
// To fit exactly into isize, we use in1 intervals of with iinner1 = iinner0 + 1
// and (idim-in1) intervals of with iinner0 
//
//         Originally:     And as result of the domain decomposition into idim subdomains:
// isize = imax-istart+1 = 2*b1 + in1*iinner1 + (idim-in1)*inner0
//
// computing is:ie, ks:ke
//   - input:            istart, imax, b1, idim, icoord (and k...)
//   - to be calculated: is, ie, iinner, iouter
//   - helper variables: iinner0, in1, isize

  isize = imax - istart + 1; // total number of elements, including the "2*b1" boundary elements
  // isize - 2*b1 = total number of unknowns
  iinner0 = (isize - 2*b1)  / idim; // smaller inner size through divide with rounding off
  in1 = isize - 2*b1 - idim * iinner0; // number of processes that must have "inner0+1" unknowns
  if (icoord < in1) {  // the first in1 processes will have "iinner0+1" unknowns
    iinner = iinner0 + 1;
    is = (istart+b1) + icoord * iinner - b1; // note that "is" reflects the position of the 
                                             // first halo or boundary element of the subdomain
  } else {             // and all other processes will have iinner0 unknowns
    iinner = iinner0;
    is = istart + in1 * (iinner0+1) + (icoord-in1) * iinner;
  }
  iouter = iinner + 2*b1;
  ie = is + iouter - 1;

  if(icoord==0)
  {
    printf("\nPlease control whether isize and the sum are identical:\n");
    printf("  isize=%3d idim=%3d || in1*(iinner0+1)=%3d *%3d + (idim-in1)*iinner0=%3d *%3d + 2*b1=2*%1d || sum = %3d\n",
              isize,    idim,       in1, iinner0+1,             idim-in1, iinner0,             b1,
                                                                           in1*(iinner0+1) + (idim-in1)*iinner0 + 2*b1);
    printf("\nPlease control whether the indices of unkowns are between %d..%d, complete, and non-overlapping:\n",
                                                                 istart+b1, imax-b1);
  }

  if(iinner>0)
    printf("  icoord=%i, iouter=%2d, iinner=%2d, subarray indices= %2d..%2d, indices of the unknowns= %2d..%2d\n",
              icoord,    iouter,     iinner  ,                      is, ie,                         is+b1, ie-b1);
  else
    printf("  icoord=%i, iouter=%2d, iinner=%2d, no subarray\n",
              icoord,    iouter,     iinner );


 } // end for(icoord...)

} // end if(myrank==0)

  MPI_Finalize();

}

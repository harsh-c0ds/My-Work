#include <stdio.h>
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
 * Authors: Rolf Rabenseifner (HLRS)                            *
 *                                                              *
 * Contact: rabenseifner@hlrs.de                                *
 *                                                              *
 * Purpose: Calculation of pi; load balancing tests             *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define f(A) (4.0/(1.0+A*A))

const int n = 10000000;

int main(int argc, char** argv)
{
  int i;
  double w, x, sum, pi;
  double wt1, wt2;

  MPI_Init(&argc, &argv);
  wt1=MPI_Wtime();
 
/* calculate pi = integral [0..1] 4/(1+x**2) dx */
  w=1.0/n;
  sum=0.0;
  for (i=0;i<n;i++)
  {
    x=w*((double)i+0.5);
    sum=sum+f(x);
  }

  wt2=MPI_Wtime();

  pi=w*sum;
  printf ("computed pi = %24.16g \n", pi);
  printf( "wall clock time = %12.4g sec\n", wt2-wt1 );

  MPI_Finalize();
  return 0;
}

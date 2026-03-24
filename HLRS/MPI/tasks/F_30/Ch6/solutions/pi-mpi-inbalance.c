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
  int my_rank, num_procs;
  int i, sub_n, sub_start;
  double w, x, sum, p_sum, pi;
  double wt1, wt2;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  // Calculating the number of elements of my subdomain: sub_n
  // Calculating the start index sub_start within 0..n-1 
  // or sub_start = -1 and sub_n = 0 if there is no element

  // The following algorithm divides 5 into 2 + 1 + 1 + 1
  sub_n = n / num_procs; // = rounding_off(n/num_procs)
  int num_larger_procs = n - num_procs*sub_n; // = #procs with sub_n+1 elements
  if (my_rank < num_larger_procs)
  { sub_n = sub_n + 1;
    sub_start = 0 + my_rank * sub_n;
  }else if (sub_n > 0)
  { sub_start = 0 + num_larger_procs + my_rank * sub_n;
  }else 
  { // this process has only zero elements
    sub_start = -1;
    sub_n = 0;
  }
  if (num_procs >= 2)
  {
    if (my_rank == num_procs-2 && sub_start >= 0)
    { // taking all remaining iterations
      sub_n = n - sub_start;
    }
    if (my_rank == num_procs-1)
    { // taking zero remaining iterations
      sub_start = -1; sub_n = 0;
    }
  }

  wt1=MPI_Wtime();
 
/* calculate pi = integral [0..1] 4/(1+x**2) dx */
  w=1.0/n;
  p_sum=0.0;
  for (i=sub_start;i<sub_start+sub_n;i++)
  {
    x=w*((double)i+0.5);
    p_sum=p_sum+f(x);
  }

  wt2=MPI_Wtime();

  MPI_Reduce (&p_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); 
  printf ("PE%i/%i: sub_n= %i,  wall clock time = %12.4g sec\n",  my_rank, num_procs, sub_n, wt2-wt1 ); 

  if (my_rank == 0) 
  { pi=w*sum;
    printf ("PE%i/%i: computed pi = %24.16g \n", my_rank, num_procs, pi);
    printf( "wall clock time = %12.4g sec\n", wt2-wt1 );
  }

  MPI_Finalize();
  return 0;
}

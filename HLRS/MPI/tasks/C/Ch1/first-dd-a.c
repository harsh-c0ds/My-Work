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
 * Authors: Joel Malard, Alan Simpson, (EPCC)                   *
 *          Rolf Rabenseifner          (HLRS)                   *
 *                                                              *
 * Contact: rabenseifner@hlrs.de                                *
 *                                                              *
 * Purpose: A first MPI example calculating the subdomain size  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

#include <stdio.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int n;  double result;  // application-related data
  int my_rank, num_procs; // MPI-related data
  int sub_n, sub_start;   // size and starting index of "my" sub domain

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  if (my_rank == 0)   
  { // reading the application data "n" from stdin only by process 0:
    printf("Enter the number of elements (n): \n");
    scanf("%d",&n); 
  }
  // broadcasting the content of variable "n" in process 0 
  // into variables "n" in all other processes:
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // Calculating the number of elements of my subdomain: sub_n
  // Calculating the start index sub_start within 0..n-1 
  // or sub_start = -1 and sub_n = 0 if there is no element

  // The following algorithm divided 5 into 2 + 2 + 1 + 0
  sub_n = (n-1) / num_procs +1; // = rounding_up(n/num_procs)
  sub_start = 0 + my_rank * sub_n;
  if (sub_start < n)
  { // this process has a real element
    if (sub_start+sub_n-1 > n-1) 
    { // but element must be smaller than sub_n
      sub_n = n - sub_start;
    } // else sub_n is already correct
  } else 
  { // this process has only zero elements
    sub_start = -1;
    sub_n = 0;
  }

  printf("I am process %i out of %i, responsible for the %i elements with indexes %2i .. %2i \n",
                       my_rank,  num_procs, sub_n, sub_start, sub_start+sub_n-1 );

  MPI_Finalize();
}

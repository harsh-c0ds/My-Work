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
 * Purpose: Gathering data from all processes, MPI_IN_PLACE     *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>


int main(int argc, char *argv[])
{
  int n;  double result;  // application-related data
  double *result_array;
  int my_rank, num_procs, rank, root; // MPI-related data

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  root = num_procs - 1;

  // doing some application work in each process, e.g.:
  result = 100.0 + 1.0 * my_rank;
  printf("I am process %i out of %i, result=%f \n", 
                       my_rank, num_procs, result);
  if (my_rank == root)
  { result_array = malloc(sizeof(double) * num_procs);
  }

  if (my_rank == root)
  {  result_array[root] = result;
     MPI_Gather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL, result_array,1,MPI_DOUBLE, root, MPI_COMM_WORLD);
  } else
  {  MPI_Gather(&result,1,MPI_DOUBLE, result_array,1,MPI_DOUBLE, root, MPI_COMM_WORLD);
  }
  if (my_rank == root)
  { for (rank=0; rank<num_procs; rank++)
      printf("I'm proc %i: result of process %i is %f \n", root, rank, result_array[rank]); 
  }

  MPI_Finalize();
}

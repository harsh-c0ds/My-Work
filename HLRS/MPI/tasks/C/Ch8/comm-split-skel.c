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
 * Authors: Joel Malard, Alan Simpson,            (EPCC)        *
 *          Rolf Rabenseifner, Traugott Streicher (HLRS)        *
 *                                                              *
 * Contact: rabenseifner@hlrs.de                                * 
 *                                                              *  
 * Purpose: A program to try MPI_Comm_split                     *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_world_rank, world_size;
  int my_sub_rank, sub_size;
  int sumA, sumB;
  int mycolor;
  MPI_Comm  sub_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);

  /* Compute sum of all ranks. */
  MPI_Allreduce (&my_world_rank, &sumA, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
  MPI_Allreduce (&my_world_rank, &sumB, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 

  printf ("PE world:%3i, color=%i sub:%3i, SumA=%3i, SumB=%3i in WORLD \n", 
          my_world_rank, mycolor, my_sub_rank, sumA, sumB);

  MPI_Finalize();
}

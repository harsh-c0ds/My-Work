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
 * Purpose: A WRONG ring communication program                  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_rank, size;
  int buffer;
  int right, left;
  int sum, i;
  MPI_Status  status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  right = (my_rank+1)      % size;
  left  = (my_rank-1+size) % size;
/* ... this SPMD-style neighbor computation with modulo has the same meaning as: */
/* right = my_rank + 1;          */
/* if (right == size) right = 0; */
/* left = my_rank - 1;           */
/* if (left == -1) left = size-1;*/

  sum = 0;
  buffer = my_rank;
  for( i = 0; i < size; i++) 
  {
    MPI_Send(&buffer, 1, MPI_INT, right, 17, MPI_COMM_WORLD);
    // WRONG program, because if MPI_Send is implemented with a
    // synchronous communication protocol then this program will deadlock!
    MPI_Recv(&buffer, 1, MPI_INT, left,  17, MPI_COMM_WORLD, &status);
    // buffer = buffer; // With only one buffer, one does not need this step 4 
    sum += buffer;
  }
  printf ("PE%i:\tSum = %i\n", my_rank, sum);

  MPI_Finalize();
}

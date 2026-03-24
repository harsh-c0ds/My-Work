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
 * Purpose: A program to try MPI_Comm_create                    *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_world_rank, world_size, my_sub_rank, sub_size;
  int sumA, sumB;
  int mycolor;
  int ranges[1][3];
  MPI_Group world_group, sub_group;
  MPI_Comm  sub_comm;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);

  mycolor = (my_world_rank > (world_size-1)/3);
  /* This definition of mycolor implies that the first color is 0 */

  /* instead of  MPI_Comm_split(MPI_COMM_WORLD, mycolor, 0, &sub_comm); 
     ... the following code is used: */

  MPI_Comm______(MPI_COMM_WORLD, &world_group);

  if (mycolor == 0) {
    /* first rank of my range:*/ ranges[0][0] = 0;
    /* last  rank of my range:*/ ranges[0][1] = (world_size-1)/3;
  } else {
    /* first rank of my range:*/ ranges[0][0] = _____;
    /* last  rank of my range:*/ ranges[0][1] = _____;
  }
  /* stride of ranks:       */ ranges[0][2] = 1;
/*
  printf ("PE world:%3i, color=%i first=%i, last=%i, stride=%i\n", 
          my_world_rank, mycolor, ranges[0][0], ranges[0][1], ranges[0][2]);
*/
  
  MPI_Group____________( ___________, _, ______, __________);
  MPI_Comm_______(MPI_COMM_WORLD, _________, &sub_comm);

  MPI_Comm_size(sub_comm, &sub_size);
  MPI_Comm_rank(sub_comm, &my_sub_rank);

  /* Compute sum of all ranks. */
  MPI_Allreduce (&my_world_rank, &sumA, 1, MPI_INT, MPI_SUM, sub_comm); 
  MPI_Allreduce (&my_sub_rank,   &sumB, 1, MPI_INT, MPI_SUM, sub_comm); 

  printf ("PE world:%3i, color=%i sub:%3i, SumA=%3i, SumB=%3i in sub_comm\n", 
          my_world_rank, mycolor, my_sub_rank, sumA, sumB);

  MPI_Finalize();
}

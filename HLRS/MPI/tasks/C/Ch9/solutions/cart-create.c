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
 * Purpose: Creating a 1-dimens. topology with MPI_Cart_create  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_rank, size;
  int snd_buf, rcv_buf;
  int right, left;
  int sum, i;

  MPI_Comm    new_comm;
  int         dims[1],
              periods[1],
              reorder;
/*int         my_coords[1]; */

  MPI_Status  status;
  MPI_Request request;

  MPI_Init(&argc, &argv);
  /* Get process info. */
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Prepare input arguments for creating a Cartesian topology. */
  dims[0] = size;
  periods[0] = 1;
  reorder = 1;
 
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &new_comm);
  /* Get reordered my_rank (and coords if ndims>1) */
  MPI_Comm_rank(new_comm, &my_rank);
  /* MPI_Cart_coords(new_comm, my_rank, 1, my_coords); */ 

  /* Get nearest neighbour rank. */
  right = (my_rank+1)      % size;
  left  = (my_rank-1+size) % size;

  /* The halo ring communication code from course chapter 4 */
  sum = 0;
  snd_buf = my_rank;
  for( i = 0; i < size; i++) 
  {
    MPI_Issend(&snd_buf, 1, MPI_INT, right, 17, new_comm, &request);
    MPI_Recv  (&rcv_buf, 1, MPI_INT, left,  17, new_comm, &status);
    MPI_Wait(&request, &status);
    snd_buf = rcv_buf;
    sum += rcv_buf;
  }
  printf ("PE%i:\tSum = %i\n", my_rank, sum);
  /* printf ("PE%i, Coords = %i: Sum = %i\n", my_rank, my_coords[0], sum); */

  MPI_Finalize();
}

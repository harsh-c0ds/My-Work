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
 * Purpose: A program with derived datatypes.                   *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_rank, size;
  int i, right, left;

  struct buff{
     int   i;
     int   j;
  } snd_buf, rcv_buf, sum;

  //__________ send_recv_type;

  MPI_Status  status;
  MPI_Request request;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  right = (my_rank+1)      % size;
  left  = (my_rank-1+size) % size;

  sum.i = 0;            sum.j = 0;
  snd_buf.i = my_rank;  snd_buf.j = 10*my_rank; // Step 1 = init

  for( i = 0; i < size; i++) 
  {
    MPI_Issend(&snd_buf, 2, MPI_INT, right, 17, MPI_COMM_WORLD, &request); // Step 2a
    MPI_Recv(&rcv_buf, 2, MPI_INT, left, 17, MPI_COMM_WORLD, &status);     // Step 3
    MPI_Wait(&request, &status);                                                  // Step 2b
    snd_buf = rcv_buf;                       // Step 4
    sum.i += rcv_buf.i;  sum.j += rcv_buf.j; // Step 5
  }

  printf ("PE%i:\tSum = %i\t%i\n", my_rank, sum.i, sum.j);

  MPI_Finalize();
}

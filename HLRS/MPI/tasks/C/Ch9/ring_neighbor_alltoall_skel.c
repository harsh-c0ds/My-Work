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
 * Purpose: Testing MPI_Neighbor_alltoall                       *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_rank, size;
  int snd_buf_arr[2], rcv_buf_arr[2];
   /* snd_buf = snd_buf_arr[0],  rcv_buf = rcv_buf_arr[0] */ // ToDo ...

  int right, left;
  int sum, i;
  MPI_Comm    new_comm;
  int         dims[1], periods[1], reorder;
  MPI_Status  status;
  MPI_Request request;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  dims[0] = size; periods[0] = 1; reorder = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, &new_comm);
  MPI_Comm_rank(new_comm, &my_rank);
  MPI_Cart_shift(new_comm, 0, 1, &left, &right);
 
  sum = 0;
  snd_buf_arr[0] = my_rank; // ToDo ...
  // snd_buf_arr[?] = -1000-my_rank; /* should be never used, only for test purpose */ // ToDo ...

  for( i = 0; i < size; i++) 
  {
    MPI_Issend(&snd_buf_arr[0], 1, MPI_INT, right, 17, new_comm, &request); // ToDo ...
    MPI_Recv(&rcv_buf_arr[0], 1, MPI_INT, left, 17, new_comm, &status); // ToDo ...
    MPI_Wait(&request, &status); // ToDo ...
    // MPI_Neighbor_alltoall(__________________________________________________________); // ToDo ...
    
    snd_buf_arr[0] = rcv_buf_arr[0]; // ToDo ...
    sum += rcv_buf_arr[0];
  }

  printf ("PE%i:\tSum = %i\n", my_rank, sum);

  MPI_Finalize();
}

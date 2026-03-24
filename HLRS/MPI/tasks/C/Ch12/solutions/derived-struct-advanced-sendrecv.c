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
 * Purpose: A program to try MPI_Sendrecv.                      *
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
     float f;
  } snd_buf, rcv_buf, sum;

  int          array_of_blocklengths[2];
  MPI_Aint     array_of_displacements[2], first_var_address, second_var_address;
  MPI_Datatype array_of_types[2], send_recv_type;

  MPI_Status  status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  right = (my_rank+1)      % size;
  left  = (my_rank-1+size) % size;

  /* Set MPI datatypes for sending and receiving partial sums. */
  array_of_blocklengths[0] = 1;
  array_of_blocklengths[1] = 1;

  MPI_Get_address(&snd_buf.i, &first_var_address);
  MPI_Get_address(&snd_buf.f, &second_var_address);

  array_of_displacements[0] = (MPI_Aint) 0;
  array_of_displacements[1] = MPI_Aint_diff(second_var_address, first_var_address);

  array_of_types[0] = MPI_INT;
  array_of_types[1] = MPI_FLOAT;

  MPI_Type_create_struct(2, array_of_blocklengths, array_of_displacements, array_of_types, &send_recv_type);
  MPI_Type_commit(&send_recv_type);

  sum.i = 0;            sum.f = 0;
  snd_buf.i = my_rank;  snd_buf.f = 10*my_rank; // Step 1 = init

  for( i = 0; i < size; i++) 
  {
    MPI_Sendrecv(&snd_buf, 1, send_recv_type, right, 17, // Step 2
                 &rcv_buf, 1, send_recv_type, left, 17,  // Step 3
                 MPI_COMM_WORLD, &status);
    snd_buf = rcv_buf;                       // Step 4
    sum.i += rcv_buf.i;  sum.f += rcv_buf.f; // Step 5
  }

  printf ("PE%i:\tSum = %i\t%f\n", my_rank, sum.i, sum.f);

  MPI_Finalize();
}

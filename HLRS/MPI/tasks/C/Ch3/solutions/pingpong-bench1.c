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
 * Purpose: A program to try MPI_Ssend and MPI_Recv.            *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

#define number_of_messages 50

int main(int argc, char *argv[])
{
  int i, my_rank;
  float buffer[1];
  double start, finish, msg_transfer_time;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  if (my_rank == 0)
  {
    MPI_Send(buffer, 1, MPI_FLOAT, 1, 17, MPI_COMM_WORLD);
    MPI_Recv(buffer, 1, MPI_FLOAT, 1, 23, MPI_COMM_WORLD, &status);
  }
  else if (my_rank == 1)
  {
    MPI_Recv(buffer, 1, MPI_FLOAT, 0, 17, MPI_COMM_WORLD, &status);
    MPI_Send(buffer, 1, MPI_FLOAT, 0, 23, MPI_COMM_WORLD);
  }

  start = MPI_Wtime();
  for (i = 1; i <= number_of_messages; i++)
  {
    if (my_rank == 0)
    {
      MPI_Send(buffer, 1, MPI_FLOAT, 1, 17, MPI_COMM_WORLD);
      MPI_Recv(buffer, 1, MPI_FLOAT, 1, 23, MPI_COMM_WORLD, &status);
    }
    else if (my_rank == 1)
    {
      MPI_Recv(buffer, 1, MPI_FLOAT, 0, 17, MPI_COMM_WORLD, &status);
      MPI_Send(buffer, 1, MPI_FLOAT, 0, 23, MPI_COMM_WORLD);
    }
  }
  finish = MPI_Wtime();

  if (my_rank == 0)
  {
    msg_transfer_time = ((finish - start) / (2 * number_of_messages)) * 1e6 ; // in microsec
    printf("Time for one messsage: %f micro seconds.\n", msg_transfer_time);
  }

  MPI_Finalize();
}

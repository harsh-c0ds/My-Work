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
 * Purpose: A program to try out one-sided communication        *
 *          with window=rcv_buf and MPI_PUT to put              *
 *          local snd_buf value into remote window (rcv_buf).   *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>


int main (int argc, char *argv[])
{
  int my_rank_world, size_world;
  int my_rank_sm,    size_sm;
  // _________ comm_sm;
  int snd_buf;
  int *rcv_buf_ptr;
  int right, left;
  int sum, i;

  MPI_Win     win;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &size_world);

//  if (my_rank_world == 0)   // PLEASE ACTIVATE IN EXERCISE 1
//  { if (size_sm == size_world) 
//    {  printf("MPI_COMM_WORLD consists of only one shared memory region\n");
//    }else
//    { printf("MPI_COMM_WORLD is split into 2 or more shared memory islands\n");
//  } }

  right = (my_rank_world+1)            % size_world;
  left  = (my_rank_world-1+size_world) % size_world;

  /* Allocate the window. */
  MPI_Win_allocate((MPI_Aint) sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &rcv_buf_ptr, &win);

  sum = 0;
  snd_buf = my_rank_world;

  for( i = 0; i < size_world; i++) 
  {
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, win);
    MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win);
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);
    
    snd_buf = *rcv_buf_ptr;
    sum += *rcv_buf_ptr;
  }

  printf ("World: %i of %i \tcomm_sm: %i of %i \tSum = %i\n", 
          my_rank_world, size_world, my_rank_sm, size_sm, sum);

  MPI_Win_free(&win);

  MPI_Finalize();
}

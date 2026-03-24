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
 * Purpose: Trying to measure whether progress may require      *
 *          the call of an unspecific MPI routines in another   *
 *          process                                             *
 *                                                              *
 * One may modify the message count by compiling, e.g., with    *
 *          mpicc -D BUF_CNT_SIZE=10000 progress-test-bsend.c   *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>
/* for malloc(): */
#include <stdlib.h>
/* for sleep(): */
#include <unistd.h>

#ifndef BUF_CNT_SIZE 
# define BUF_CNT_SIZE 1000000
#endif

int main (int argc, char *argv[])
{
  int my_rank, size;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* To be sure that all processes are started. */
  MPI_Barrier(MPI_COMM_WORLD); 
  sleep(1); // to be sure that all process left the MPI_Barrier

  if (my_rank == 0) /* buffered sender */ 
  {
      float *snd_buf;
      char *bsend_buffer;
      int bsend_buffer_size;
      double time_begin, time_end;

      printf("\nMessage size is %ld bytes \n\n", (long)(BUF_CNT_SIZE * sizeof(float)));

      snd_buf = (float *)malloc(BUF_CNT_SIZE * sizeof(float));
      MPI_Pack_size(BUF_CNT_SIZE, MPI_FLOAT, MPI_COMM_WORLD, &bsend_buffer_size);      
      bsend_buffer_size = bsend_buffer_size + MPI_BSEND_OVERHEAD;
      bsend_buffer = (char *)malloc(bsend_buffer_size);
      MPI_Buffer_attach(bsend_buffer, bsend_buffer_size);

      printf("I am %i before bsend call \n", my_rank);
      time_begin = MPI_Wtime();
      MPI_Bsend(snd_buf, BUF_CNT_SIZE, MPI_FLOAT, 1, 17, MPI_COMM_WORLD);
      time_end = MPI_Wtime();
      printf("I am %i after bsend call, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
      printf("I am %i before sleep(40) \n", my_rank);
      time_begin = MPI_Wtime();
      sleep(40); 
      time_end = MPI_Wtime();
      printf("I am %i after sleep(40), delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
      printf("I am %i before barrier call \n", my_rank);
      time_begin = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD); // as an unspecific MPI call that allows to complete the internal work for the (weak) local MPI_Bsend
      time_end = MPI_Wtime();
      printf("I am %i after barrier call, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
  }
  else if (my_rank == 1) /* delayed receiver */
  {
      float *rcv_buf;
      double time_begin, time_end;

      rcv_buf = (float *)malloc(BUF_CNT_SIZE * sizeof(float));
   
      printf("                              I am %i before sleep(15) \n", my_rank);
      time_begin = MPI_Wtime();
      sleep(15); // 15 seconds ago, process 0 has Bsend the message
      time_end = MPI_Wtime();
      printf("                              I am %i after sleep(15), delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
      printf("                              I am %i: 15 seconds ago, process 0 called MPI_Bsend \n", my_rank );
      printf("                              I am %i before recv call \n", my_rank);
      printf("                              I am %i: With progress, the following MPI_Recv will come back after a few micro seconds \n", my_rank );
      printf("                              I am %i: Without progress, the following MPI_Recv will wait another 25 seconds until process 0 will call the barrier = unspecific call following the Bsend \n", my_rank );
      time_begin = MPI_Wtime();
      MPI_Recv(rcv_buf, BUF_CNT_SIZE, MPI_FLOAT, 0, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      time_end = MPI_Wtime();
      printf("                              I am %i after recv call, delta time = %9.6lf sec \n", my_rank, time_end-time_begin );
      printf("                              I am %i before barrier call \n", my_rank);
      printf("                              I am %i: With progress, the previous MPI_Recv was instantly done and the barrier has to wait for the other 25 sec \n", my_rank );
      printf("                              I am %i: Without progress, the Recv waited for the other 25 sec and therefore the following barrier needs only a few micro seconds \n", my_rank );
      time_begin = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD); // as an unspecific MPI call that allows to complete the internal work for the (weak) local MPI_Bsend in the other process
      time_end = MPI_Wtime();
      printf("                              I am %i after barrier call, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
  }
  else /* other processes that are not involved */
  {
      MPI_Barrier(MPI_COMM_WORLD); // that all processes have called the barrier
  }

  MPI_Finalize();
}

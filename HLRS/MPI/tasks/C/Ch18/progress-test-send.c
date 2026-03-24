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
 *          mpicc -D BUF_CNT_SIZE=10000 progress-test-send.c    *
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

  if (my_rank == 0) /* standard send sender */ 
  {
      int token;
      float *snd_buf;
      double time_begin, time_end;

      snd_buf = (float *)malloc(BUF_CNT_SIZE * sizeof(float));

      time_begin = MPI_Wtime();
      MPI_Recv(&token, 1, MPI_INT, 1, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      time_end = MPI_Wtime();
      printf("I am %i token received, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
   
      printf("I am %i before sleep(5) \n", my_rank);
      time_begin = MPI_Wtime();
      sleep(5); // 5 seconds sleeping after preparation
      time_end = MPI_Wtime();
      printf("I am %i after sleep(5), delta = time %9.6lf sec \n", my_rank, time_end-time_begin );

      printf("I am %i before Send call. This call is expected to be local because the receive (as an MPI_Irecv) is already called in process 1\n", my_rank);
      printf("I am %i: With progress, the following MPI_Send will come back after a few micro seconds \n", my_rank );
      printf("I am %i: Without progress, the following MPI_Send will wait another 15 seconds until process 1 will call the barrier = unspecific call following the Irecv \n", my_rank );
      time_begin = MPI_Wtime();
      MPI_Send(snd_buf, BUF_CNT_SIZE, MPI_FLOAT, 1, 17, MPI_COMM_WORLD);
      time_end = MPI_Wtime();
      printf("I am %i after Send call, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );

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
      int token;
      MPI_Request rq;
      float *rcv_buf;
      double time_begin, time_end;

      rcv_buf = (float *)malloc(BUF_CNT_SIZE * sizeof(float));

      time_begin = MPI_Wtime();
      MPI_Irecv(rcv_buf, BUF_CNT_SIZE, MPI_FLOAT, 0, 17, MPI_COMM_WORLD, &rq);
      MPI_Send(&token, 1, MPI_INT, 0, 13, MPI_COMM_WORLD);
      time_end = MPI_Wtime();
      printf("                              I am %i Irecv started and token sent, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
   
      printf("                              I am %i before sleep(5) \n", my_rank);
      time_begin = MPI_Wtime();
      sleep(5); // 5 seconds sleeping after preparation
      time_end = MPI_Wtime();
      printf("                              I am %i after sleep(5), delta = time %9.6lf sec \n", my_rank, time_end-time_begin );

      printf("                              I am %i before sleep(15) \n", my_rank);
      time_begin = MPI_Wtime();
      sleep(15); // 15 seconds ago, process 0 has Bsend the message
      time_end = MPI_Wtime();
      printf("                              I am %i after sleep(15), delta = time %9.6lf sec \n", my_rank, time_end-time_begin );
      printf("                              I am %i: 15 seconds ago, process 0 called MPI_Send \n", my_rank );

      printf("                              I am %i before barrier call \n", my_rank);
      printf("                              I am %i: With MPI_Send progress in process 0, this barrier has to wait for the barrier in proces 0 coming in 40-15 = 25 sec \n", my_rank );
      printf("                              I am %i: Without MPI_Send progress in process 0, the Send waited for the 15 sec and therefore the following barrier ihas to wait 40 sec \n", my_rank );
      time_begin = MPI_Wtime();
      MPI_Barrier(MPI_COMM_WORLD); // as an unspecific MPI call that allows to complete the internal work for the (weak) local MPI_Bsend in the other process
      time_end = MPI_Wtime();
      printf("                              I am %i after barrier call, delta = time %9.6lf sec \n", my_rank, time_end-time_begin );

      printf("                              I am %i before recv-wait call \n", my_rank);
      time_begin = MPI_Wtime();
      MPI_Wait(&rq, MPI_STATUS_IGNORE);
      time_end = MPI_Wtime();
      printf("                              I am %i after recv-wait call, delta time = %9.6lf sec \n", my_rank, time_end-time_begin );
  }
  else /* other processes that are not involved */
  {
      MPI_Barrier(MPI_COMM_WORLD); // that all processes have called the barrier
  }

  MPI_Finalize();
}

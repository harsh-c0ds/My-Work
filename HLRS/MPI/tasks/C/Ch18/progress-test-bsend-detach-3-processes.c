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
 * Authors: Rolf Rabenseifner, Traugott Streicher (HLRS)        *
 *                                                              *
 * Contact: rabenseifner@hlrs.de                                * 
 *                                                              *  
 * Purpose: Trying to measure whether progress may require      *
 *          the call of an unspecific MPI routines in another   *
 *          process                                             *
 *          This program tries to resolv the bsend problem      *
 *          by adding a buffer detach (which must wait)         *
 *          plus a re-attach directly after the receive.        *
 *                                                              *
 * progress-test-bsend-detach-3-processes.c                     *
 * uses MPI_Bsend + MPI_Recv + detach + attach and shows that   *
 * the progress problem of using only MPI_Bsend + MPI_Recv in   *
 * progress-test-bsend-3-processes.c is resolved.               *
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
/* for strcpy(): */
#include <string.h>

#ifndef BUF_CNT_SIZE 
# define BUF_CNT_SIZE 100000
#endif

int main (int argc, char *argv[])
{
  int my_rank, size, experiment, iter, delay;
  float *snd_buf, *rcv_buf;
  char *bsend_buffer;
  int bsend_buffer_size;
  double time_begin;
  char indent[200];

#ifdef USE_THREAD_MULTIPLE
  int thread_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &thread_provided);
#else
  MPI_Init(&argc, &argv);
#endif

  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (my_rank == 0) {
    printf("\n");
    printf("Problems with MPI_Bsend and the MPI progress implementations\n");
    printf("============================================================\n");
    printf("\n");
    printf("Same as in progress-test-bsend-3-processes.c, but the progress problem\n");
    printf("is resolved by using Bsend + Recv + Buffer_detach + Buffer_attach.\n");
    printf("\n");
    printf("Compile-time options:\n");
    printf("=====================\n");
    printf("\n");
    printf("  -D BUF_CNT_SIZE=2000          to use less or more than 100000 floats as message size\n");
    printf("  -D USE_THREAD_MULTIPLE        to use MPI_Init_thread(MPI_THREAD_MULTIPLE instead of MPI_Init\n");
    printf("\n");
    printf("Used compile-time options:\n");
    printf("==========================\n");
    printf("\n");
    printf("For all experiments, BUF_CNT_SIZE=%1d ==> the message size is %ld bytes \n\n", 
                                 BUF_CNT_SIZE, (long)(BUF_CNT_SIZE * sizeof(float)));
    printf("\n");
#ifdef USE_THREAD_MULTIPLE
  if (my_rank == 0) {
      printf("MPI_Init_thread(MPI_THREAD_MULTIPLE,...) returned  provided= %s\n\n",
       (thread_provided == MPI_THREAD_SINGLE     ? "MPI_THREAD_SINGLE"     :
        (thread_provided == MPI_THREAD_FUNNELED   ? "MPI_THREAD_FUNNELED"   :
         (thread_provided == MPI_THREAD_SERIALIZED ? "MPI_THREAD_SERIALIZED" :
          (thread_provided == MPI_THREAD_MULTIPLE   ? "MPI_THREAD_MULTIPLE"   : "MPI_THREAD_ unknown"
      )))));
  }
  if (thread_provided != MPI_THREAD_MULTIPLE) {
    printf("\nApplication aborted, because MPI_THREAD_MULTIPLE is not provided\n");
    MPI_Finalize();
    exit(0);
  }
#else
    printf("MPI_Init is used\n");
#endif
    printf("\n");
    printf("Run-time options:\n");
    printf("=====================\n");
    printf("\n");
    printf("export MPICH_ASYNC_PROGRESS=1   only for mpich to switch on an asynchronous progress thread\n");
    printf("\n");
    printf("\n");
    printf("And here is the general pattern:\n");
    printf("================================\n");
    printf("\n");
    printf("The experiments are designed for 3 processes, but can also be used with more than 3 processes.\n");
    printf("In that case, the pattern of process rank==1 is used for all \"middle\" processes\n");
    printf("\n");
    printf("After a barrier, the following pattern is implemented in a loop of three iterations:\n");
    printf("\n");
    printf("         myrank==0             myrank==1             myrank==2       \n");
    printf("                            bsend(to:myrank-1)    bsend(to:myrank-1) \n");
    printf("      bsend(to:myrank+1)    bsend(to:myrank+1)                       \n");
    printf("                            recv(from:myrank-1)   recv(from:myrank-1)\n");
    printf("      recv(from:myrank+1)   recv(from:myrank+1)                      \n");
    printf("       20 sec numerics       20 sec numerics       20 sec numerics   \n");
    printf("\n");
    printf("In our 1st experiment, the pattern is a bit disturbed by some delays between the bsend and the recv:\n");
    printf("=====================\n");
    printf("\n");
    printf("iter     myrank==0             myrank==1             myrank==2       \n");
    printf("  time\n");
    printf("1  0                        bsend(to:myrank-1)    bsend(to:myrank-1) \n");
    printf("1  0  bsend(to:myrank+1)    bsend(to:myrank+1)                       \n");
    printf("1  0     sleep(5)              sleep(3)              sleep(1)        marking some different delays\n");
    printf("1  1                                              recv(from:myrank-1)\n");
    printf("1  1                                                 sleep(20)       marking some balanced numerics\n");
    printf("1  3                        recv(from:myrank-1)                      \n");
    printf("1  3                        recv(from:myrank+1)                      \n");
    printf("1  3                           sleep(20)                             marking some balanced numerics\n");
    printf("1  5  recv(from:myrank+1)                                            \n");
    printf("1  5     sleep(20)                                                   marking some balanced numerics\n");
    printf("\n");
    printf("The following listing shows all three iterations if asynchronous progress properly works\n");
    printf("\n");
    printf("iter     myrank==0             myrank==1             myrank==2       \n");
    printf("  time\n");
    printf("1  0     barrier               barrier               barrier         \n");
    printf("1  0                        bsend(to:myrank-1)    bsend(to:myrank-1) \n");
    printf("1  0  bsend(to:myrank+1)    bsend(to:myrank+1)                       \n");
    printf("1  0     sleep(5)              sleep(3)              sleep(1)        marking some different delays\n");
    printf("1  1        :                     :               recv(from:myrank-1)\n");
    printf("1  1        :                     :                  sleep(20)       marking some balanced numerics\n");
    printf("1  3        :               recv(from:myrank-1)         :            \n");
    printf("1  3        :               recv(from:myrank+1)         :            \n");
    printf("1  3        :                  sleep(20)                :            marking some balanced numerics\n");
    printf("1  5  recv(from:myrank+1)         :                     :            \n");
    printf("1  5     sleep(20)                :                     :            marking some balanced numerics\n");
    printf("1  6        :                     :                     :            all in numerics\n");
    printf("1 ...                                                                all in numerics\n");
    printf("1 20        :                     :                     :            all in numerics\n");
    printf("2 21        :                     :               bsend(to:myrank-1) \n");
    printf("2 21        :                     :                  sleep(1)        marking some different delays\n");
    printf("2 22        :                     :               recv(from:myrank-1)\n");
    printf("2 22        :                     :                     |            blocked due to late sender\n");
    printf("2 23        :               bsend(to:myrank-1)          |            \n");
    printf("2 23        :               bsend(to:myrank+1)    return from recv   after 1 sec idle time\n");
    printf("2 23        :                  sleep(3)                              marking some different delays\n");
    printf("2 23        :                     :                  sleep(20)       marking some balanced numerics\n");
    printf("2 25  bsend(to:myrank+1)          :                     :            \n");
    printf("2 25     sleep(5)                 :                     :            marking some different delays\n");
    printf("2 26        :               recv(from:myrank-1)         :            \n");
    printf("2 26        :               recv(from:myrank+1)         :            \n");
    printf("2 26        :                  sleep(20)                :            marking some balanced numerics\n");
    printf("2 30  recv(from:myrank+1)         :                     :            \n");
    printf("2 30     sleep(20)                :                     :            marking some balanced numerics\n");
    printf("2 31        :                     :                     :            all in numerics\n");
    printf("2 ...                                                                all in numerics\n");
    printf("2 42        :                     :                     :            all in numerics\n");
    printf("3 43        :                     :               bsend(to:myrank-1) \n");
    printf("3 43        :                     :                  sleep(1)        marking some different delays\n");
    printf("3 44        :                     :               recv(from:myrank-1)\n");
    printf("3 46        :               bsend(to:myrank-1)          |            \n");
    printf("3 46        :               bsend(to:myrank+1)    return from recv   2 sec idle time due to late sender\n");
    printf("3 46        :                  sleep(3)              sleep(20)       marking some different delays\n");
    printf("3 49        :               recv(from:myrank-1)         :            \n");
    printf("3 49        :                     |                     :            blocked due to late sender\n");
    printf("3 50  bsend(to:myrank+1)    return from recv            :            after 1 sec idle time\n");
    printf("3 50                        recv(from:myrank+1)         :            \n");
    printf("2 50     sleep(5)                                       :            marking some different delays\n");
    printf("3 50        :                  sleep(20)                :            marking some balanced numerics\n");
    printf("3 55  recv(from:myrank+1)         :                     :            \n");
    printf("3 55     sleep(20)                :                     :            marking some balanced numerics\n");
    printf("3 56        :                     :                     :            all in numerics\n");
    printf("3 ...                                                                all in numerics\n");
    printf("3 65        :                     :                     :            all in numerics\n");
    printf("3 66        :                     :                  back from sleep marking some balanced numerics\n");
    printf("  66        :                     :                  barrier         \n");
    printf("3 70        :                  back from sleep          |            marking some balanced numerics\n");
    printf("  70        :                  barrier                  |            \n");
    printf("3 75     back from sleep          |                     |            marking some balanced numerics\n");
    printf("  75     barrier                  |                     |            \n");
    printf("  75   back from barrier     back from barrier     back from barrier \n");

    printf("\n");
    printf("==========================================\n");
    printf("=== And now, let's do the experiments: ===\n");
    printf("==========================================\n");
  }
    
  snd_buf = (float *)malloc(BUF_CNT_SIZE * sizeof(float));
  rcv_buf = (float *)malloc(BUF_CNT_SIZE * sizeof(float));
  MPI_Pack_size(BUF_CNT_SIZE, MPI_FLOAT, MPI_COMM_WORLD, &bsend_buffer_size);      
// (RESOLVED: next message may be sent before first message is received on the destination process ==> *2 )
// (and) if my_rank is not 0 or size-1 then (additionally)
//       into 2 directions (my_rank-1 and my-rank+1) ==> *2, (i.e., in total *4)
  if ((my_rank == 0) || (my_rank == (size-1))) {
    bsend_buffer_size = ( bsend_buffer_size + MPI_BSEND_OVERHEAD ) * 1; // instead of 2
  }else{
    bsend_buffer_size = ( bsend_buffer_size + MPI_BSEND_OVERHEAD ) * 2; // instead of 4
  }
  bsend_buffer = (char *)malloc(bsend_buffer_size);
  MPI_Buffer_attach(bsend_buffer, bsend_buffer_size);

  for (experiment=1; experiment<=2; experiment++) 
  { 
    MPI_Barrier(MPI_COMM_WORLD); 
    if (my_rank == 0) {
      printf("\nExperiment %1d: %s\n", experiment,
       (experiment==1 ? "delays are 5, 3, and 1 seconds for my_rank = 0, 1..size-2, size-1"
                      : "all delays are 0 seconds") );
      printf(  "=============\n\n");
    }
  
    /* To be sure that all processes are started. */
    MPI_Barrier(MPI_COMM_WORLD); 
    sleep(1); // to be sure that all process left the MPI_Barrier
  
    time_begin = MPI_Wtime();
    for (iter=1; iter <=3; iter++ ) 
    {
      if (my_rank == 0)
      {
        delay = (experiment==1 ? 5 : 0);
        strcpy(indent, "");
          printf("[%1i] %1i %5.1f%s calling bsend(to:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Bsend(snd_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank+1, 17, MPI_COMM_WORLD);
          printf("[%1i] %1i %5.1f%s return  bsend(to:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling sleep(%1d)\n", my_rank, iter, MPI_Wtime()-time_begin, indent, delay);
        sleep(delay);
          printf("[%1i] %1i %5.1f%s return  sleep(%1d)\n", my_rank, iter, MPI_Wtime()-time_begin, indent, delay);
          printf("[%1i] %1i %5.1f%s calling recv (from:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Recv(rcv_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank+1, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("[%1i] %1i %5.1f%s return  recv (from:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling detach(+attach)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Buffer_detach(&bsend_buffer, &bsend_buffer_size);
        MPI_Buffer_attach(bsend_buffer, bsend_buffer_size); // unreported, because a strongly local routine
          printf("[%1i] %1i %5.1f%s return  detach(+attach)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s start numerics(20)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        sleep(20);
          printf("[%1i] %1i %5.1f%s end of numerics(20)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
      }
      else if (my_rank < size-1) /* delayed receiver */
      {
        delay = (experiment==1 ? 3 : 0);
        strcpy(indent, "   -   -   -   -   -   - ");
        for (int i=1; i<my_rank; i++) strcat(indent, "  ");
          printf("[%1i] %1i %5.1f%s calling bsend(to:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Bsend(snd_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank-1, 17, MPI_COMM_WORLD);
          printf("[%1i] %1i %5.1f%s return  bsend(to:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling bsend(to:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Bsend(snd_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank+1, 17, MPI_COMM_WORLD);
          printf("[%1i] %1i %5.1f%s return  bsend(to:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling sleep(%1d)\n", my_rank, iter, MPI_Wtime()-time_begin, indent, delay);
        sleep(delay);
          printf("[%1i] %1i %5.1f%s return  sleep(%1d)\n", my_rank, iter, MPI_Wtime()-time_begin, indent, delay);
          printf("[%1i] %1i %5.1f%s calling recv (from:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Recv(rcv_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank-1, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("[%1i] %1i %5.1f%s return  recv (from:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling recv (from:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Recv(rcv_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank+1, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("[%1i] %1i %5.1f%s return  recv (from:+1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling detach(+attach)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Buffer_detach(&bsend_buffer, &bsend_buffer_size);
        MPI_Buffer_attach(bsend_buffer, bsend_buffer_size); // unreported, because a strongly local routine
          printf("[%1i] %1i %5.1f%s return  detach(+attach)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s start numerics(20)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        sleep(20);
          printf("[%1i] %1i %5.1f%s end of numerics(20)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
      }
      else /* my_rank == size-1 */
      {
        delay = (experiment==1 ? 1 : 0);
        strcpy(indent, "   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~   ~  ");
          printf("[%1i] %1i %5.1f%s calling bsend(to:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Bsend(snd_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank-1, 17, MPI_COMM_WORLD);
          printf("[%1i] %1i %5.1f%s return  bsend(to:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling sleep(%1d)\n", my_rank, iter, MPI_Wtime()-time_begin, indent, delay);
        sleep(delay);
          printf("[%1i] %1i %5.1f%s return  sleep(%1d)\n", my_rank, iter, MPI_Wtime()-time_begin, indent, delay);
          printf("[%1i] %1i %5.1f%s calling recv (from:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Recv(rcv_buf, BUF_CNT_SIZE, MPI_FLOAT, my_rank-1, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("[%1i] %1i %5.1f%s return  recv (from:-1)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s calling detach(+attach)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        MPI_Buffer_detach(&bsend_buffer, &bsend_buffer_size);
        MPI_Buffer_attach(bsend_buffer, bsend_buffer_size); // unreported, because a strongly local routine
          printf("[%1i] %1i %5.1f%s return  detach(+attach)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
          printf("[%1i] %1i %5.1f%s start numerics(20)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
        sleep(20);
          printf("[%1i] %1i %5.1f%s end of numerics(20)\n", my_rank, iter, MPI_Wtime()-time_begin, indent);
      }
    } // for iter

      printf("[%1i] - %5.1f%s calling barrier\n", my_rank, MPI_Wtime()-time_begin, indent);
    MPI_Barrier(MPI_COMM_WORLD); // that all processes have called the barrier
      printf("[%1i] - %5.1f%s return  barrier\n", my_rank, MPI_Wtime()-time_begin, indent);

  } // for experiment

  MPI_Finalize();
}

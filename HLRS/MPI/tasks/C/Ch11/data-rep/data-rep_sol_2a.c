/* ************************************************************************** */
/*                                                                            */
/* data-replication in distributed and shared memory                          */
/* program (C source code).                                                   */
/*                                                                            */
/* - the skeleton bcasts the data to all processes                            */
/* - solution: rank_world == 0 puts the data                                  */
/*             into the shared memory of node 0 ,                             */
/*             rank_world == 0 bcasts the data to one of the processes        */
/*             of each of the other nodes, only ,                             */
/*             i.e., to all the other rank_shm==0 processes                   */
/*                                                                            */
/* - Course material: Introduction to Hybrid Programming in HPC               */
/*                                                                            */
/*                    It is made freely available with the understanding that */
/*                    every copy must include this header and that            */
/*                    the authors as well as VSC and TU Wien                  */
/*                    take no responsibility for the use of this program.     */
/*                                                                            */
/*        (c) 01/2019 Irene Reichl (VSC Team, TU Wien)                        */
/*                    irene.reichl@tuwien.ac.at                               */
/*                                                                            */
/*      vsc3:  module load intel/18 intel-mpi/2018                            */
/*      vsc3:  mpiicc -o data-rep_solution data-rep_solution.c                */
/*                                                                            */
/* ************************************************************************** */


#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

typedef long arrType ;
#define arrDataType MPI_LONG /* !!!!!   C A U T I O N :   MPI_Type must fit to arrType       !!!!! */
static const int arrSize=16*1.6E7 ;

int main (int argc, char *argv[])
{
  int it ;
  int rank_world, size_world;
  arrType *arr ;
  int i;
  long long sum ;

/* ===> 1 <=== */
/* TO DO --> comm_shm, size_shm, rank_shm, shared window win, *shm_buf_ptr
 * individualShmSize 
 
 * MPI_Win_shared_query --> *arr 
 
 * comm_head; size_head, rank_head;
 */
  MPI_Comm comm_shm; 
  int size_shm, rank_shm;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &size_world);

/* ===> 2 <=== */
  /* Create --> shared memory islands and --> shared memory window inside */
  /*           -->    comm_shm         and      -->    win                 */

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, /*key=*/ 0, MPI_INFO_NULL, &comm_shm);
  MPI_Comm_size(comm_shm, &size_shm); 
  MPI_Comm_rank(comm_shm, &rank_shm); 
  /*TEST*/ // To minimize the output, we print only from 3 process per SMP node 
  /*TEST*/ if ( rank_shm == 0 || rank_shm == 1 || rank_shm == size_shm - 1 )
      printf("\t\trank ( world: %i, shm: %i)\n", rank_world, rank_shm);
  /*TEST*/ if(rank_world==0) printf("ALL finalize and return !!!.\n"); MPI_Finalize(); return 0;

/*  TO DO: 
 *  substitute the following malloc
 */
    arr = (arrType *) malloc(arrSize * sizeof(arrType));
    if(arr == NULL)
    {   printf("arr NOT allocated, not enough memory\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
/*  by
 *   MPI_Comm_split_type        ---> create shm islands  (DONE !!!!!!!!)
 *   MPI_Win_allocate_shared    ---> create shm windows
 *                                   such that rank_shm = 0 has shared memory portion equal to size of arr
 *                                   and rank_shm != 0 have shared memory portion equal to zero
 *   MPI_Win_shared_query       ---> get starting address of rank_shm = 0 's shared memory portion
 */
  /* Note that buffer pointer returned from MPI_Win_allocate_shared
     is not used because it is only available in process rank_shm==0 */

/* TO DO: Create communicator comm_head with MPI_Comm_split  -->  including all the rank_shm == 0 processes.
 * Only the rank_shm==0 processes should have a color, all others have color=MPI_UNDEFINED 
 * this meands that only the head processes are in the communicator, the other processes are not included,
 * i.e.,  their comm_head is MPI_COMM_NULL.
 * The advantage of excluding all the other ranks compared to assigning color = 1 is that the communicator 
 * is a smaller entity, hence, it allows for better scaling.
 *
 * Note that subsequent MPI-calls within comm_head must only be performed for processes on comm_head
 * -->   if( comm_head != MPI_COMM_NULL ) 
 *
 * Choose key value in MPI_Comm_split_type(.... &comm_shm) and MPI_Comm_split(.... &comm_head) 
 * such that the rank_world==0 process has also rank==0 within its comm_shm and within comm_head.
 */
  
  // int color;
  // color=MPI_UNDEFINED ;
  // if (rank_shm==0) color = 0 ;
  // MPI_Comm_split(... &comm_head);

/* ===> 3 <=== */
 for( it = 0; it < 3; it++)
 {

/* TO DO: the following line writes on the shared memory.
 *        take care for correct synchronization with memory fences !   
 *        The MPI_Win_fence starts the write epoch for all rank_shm==0 processes
 */
 /* only rank_world=0 initializes the array arr                 */
   if( rank_world == 0 )
   {
     for( i = 0; i < arrSize; i++)
     { arr[i] = i + it ; }
   }

/* ===> 4 <=== */
/* TO DO: do the Bcast only in comm_head !
 */
   MPI_Bcast( arr, arrSize, arrDataType, 0, MPI_COMM_WORLD );

/* Now, all arrays are filled with the same content. */
   
/* ===> 5 <=== */
/* TO DO: the following lines read from the shared memory and compute the sum.
 *        take care for correct synchronization with memory fences !   
 *        The MPI_Win_fence now has to switch from the write epoch to the read epoch
 *        allowing reading arr by all processes belonging to their comm_shm.
 */
   sum = 0;
   for( i = 0; i < arrSize; i++)
   { 
     sum+= arr [ i ] ;
   }
  
/* ===> 6 <=== */
   printf ("it: %i, rank ( world: %i ):\tsum(i=%i...i=%i) = %lld \n",
                 it, rank_world, it, arrSize-1+it, sum );
   /* already prepared for the shared memory solution: */
   // printf ("it: %i, rank ( world: %i, shm: %i, head: %i ):\tsum(i=%d...i=%d) = %lld \n",
   //              it, rank_world, rank_shm, rank_head, it, arrSize-1+it, sum );
 }

/* ===> 7 <=== */
/* TO DO: there is no malloc and therefore no free of arr.
 *        instead free the shared memory and
 *        guarantee that all operations on the shared memory have been completed before the shared memory is freed
 */
  free(arr);
  MPI_Finalize();
}

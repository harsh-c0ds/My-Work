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
  MPI_Comm comm_shm; 
  int size_shm, rank_shm;
  MPI_Win  win;
  int individualShmSize ;
  arrType *shm_buf_ptr;

  /* output MPI_Win_shared_query */
  MPI_Aint arrSize_ ;
  int disp_unit ;

  int color ;
  MPI_Comm comm_head;
  int size_head, rank_head;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &size_world);

/* ===> 2 <=== */
  /* Create --> shared memory islands and --> shared memory window inside */
  /*           -->    comm_shm         and      -->    win                 */

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, /*key=*/ 0, MPI_INFO_NULL, &comm_shm);
  MPI_Comm_size(comm_shm, &size_shm); 
  MPI_Comm_rank(comm_shm, &rank_shm); 

  /* instead of:  arr = (arrType *) malloc(arrSize * sizeof(arrType)); */
  if ( rank_shm == 0 )
  { individualShmSize = arrSize ; }
  else
  { individualShmSize = 0 ; }
  MPI_Win_allocate_shared( (MPI_Aint)(individualShmSize) * (MPI_Aint)(sizeof(arrType)), sizeof(arrType), MPI_INFO_NULL, comm_shm, &shm_buf_ptr, &win );
  /* shm_buf_ptr is not used because it is only available in process rank_shm==0 */
  MPI_Win_shared_query( win, 0, &arrSize_, &disp_unit, &arr );

  /* Create communicator including all the rank_shm = 0               */
  /* with the MPI_Comm_split: in color 0 all the rank_shm = 0 , 
   * all other ranks are color = 1                                        */

  color=MPI_UNDEFINED ;
  if (rank_shm==0) color = 0 ;

  MPI_Comm_split(MPI_COMM_WORLD, color, /*key=*/ 0, &comm_head);
  rank_head = -1; // only used in the print statements to differentiate unused rank==-1 from used rank==0
  if( comm_head != MPI_COMM_NULL ) // if( color == 0 ) // rank is element of comm_head, i.e., it is head of one of the islands in comm_shm
  {
    MPI_Comm_size(comm_head, &size_head); 
    MPI_Comm_rank(comm_head, &rank_head); 
  }

/* ===> 3 <=== */
 for( it = 0; it < 3; it++)
 {
 /* only rank_world=0 initializes the array arr                 */
 /* all rank_shm=0 start the write epoch: writing arr to their shm */
   MPI_Win_fence(/*workaround: no assertions:*/ 0, win); 
   if( rank_world == 0 ) /* from those rank_shm=0 processes, only rank_world==0 fills arr */
   {
     for( i = 0; i < arrSize; i++)
     { arr[i] = i + it ; }
   }

/* ===> 4 <=== */
 /* Instead of all processes in MPI_COMM_WORLD, now only the heads of the 
  * shared memory islands communicate (using comm_head).
  * Since we used key=0 in both MPI_Comm_split(...), process rank_world = 0
  * - is also rank 0 in comm_head
  * - and rank 0 in comm_shm in the color it belongs to.                              */

   if( comm_head != MPI_COMM_NULL ) // if( color == 0 )
   {
     MPI_Bcast(arr, arrSize, arrDataType, 0, comm_head);
     /* with this Bcast, all other rank_shm=0 processes write the data into their arr */
   }

/* Now, all arrays are filled with the same content. */

/* ===> 5 <=== */
   MPI_Win_fence(/*workaround: no assertions:*/ 0, win); // after the fence all processes start a read epoch

/* Now, all other ranks in the comm_sm shared memory islands are allowed to access their shared memory array. */
/* And all ranks rank_sm access the shared mem in order to compute sum  */
   sum = 0;
   for( i = 0; i < arrSize; i++)
   { 
     //sum+= *( shm_buf_ptr - rank_shm * shmSize + i ) ;
     sum+= arr [ i ] ;
   }
  
/* ===> 6 <=== */
  /*TEST*/ // To minimize the output, we print only from 3 process per SMP node 
  /*TEST*/ if ( rank_shm == 0 || rank_shm == 1 || rank_shm == size_shm - 1 )
     printf ("it: %i, rank ( world: %i, shm: %i, head: %i ):\tsum(i=%d...i=%d) = %lld \n",
                  it, rank_world, rank_shm, rank_head, it, arrSize-1+it, sum );
 }
  /*TEST*/ if(rank_world==0) printf("ALL finalize and return !!!.\n"); MPI_Finalize(); return 0;

/* ===> 7 <=== */
/* TO DO: there is no malloc and therefore no free of arr.
 *        instead free the shared memory and
 *        guarantee that all operations on the shared memory have been completed before the shared memory is freed
 */
  free(arr);
  MPI_Finalize();
}

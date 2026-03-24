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

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &size_world);

/* ===> 2 <=== */
    arr = (arrType *) malloc(arrSize * sizeof(arrType));
    if(arr == NULL)
    {   printf("arr NOT allocated, not enough memory\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

/* ===> 3 <=== */
 for( it = 0; it < 3; it++)
 {
 /* only rank_world=0 initializes the array arr                 */
   if( rank_world == 0 )
   {
     for( i = 0; i < arrSize; i++)
     { arr[i] = i + it ; }
   }
/* ===> 4 <=== */
   MPI_Bcast( arr, arrSize, arrDataType, 0, MPI_COMM_WORLD );

/* Now, all arrays are filled with the same content. */
   
/* ===> 5 <=== */
   sum = 0;
   for( i = 0; i < arrSize; i++)
   { 
     sum+= arr [ i ] ;
   }
  
/* ===> 6 <=== */
  /*TEST*/ // To minimize the output, we print only from 3 process per SMP node 
  /*TEST*/ if ( rank_world == 0 || rank_world == 1 || rank_world == size_world - 1 )
      printf ("it: %i, rank ( world: %i ):\tsum(i=%i...i=%i) = %lld \n",
                   it, rank_world, it, arrSize-1+it, sum );
 }

/* ===> 7 <=== */
  free(arr);
  MPI_Finalize();
}

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
  int my_rank_sm_sub,size_sm_sub;
  MPI_Comm  comm_sm, comm_sm_sub;
  int snd_buf;
  int *rcv_buf_ptr;
  int right, left;
  int sum, i;

  MPI_Win     win;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &size_world);

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_sm);
  MPI_Comm_rank(comm_sm, &my_rank_sm);
  MPI_Comm_size(comm_sm, &size_sm);
  if (my_rank_world == 0)
  { if (size_sm == size_world) 
    {  printf("MPI_COMM_WORLD consists of only one shared memory region\n");
    }else
    { printf("MPI_COMM_WORLD is split into 2 or more shared memory islands\n");
  } }

#define split_method_SPLIT
// #define split_method_OPENMPI
// #define split_method_MPICH
#if defined(split_method_SPLIT)
  { int color;
    // Splitting comm_sm into smaller sub-islands. Of course, they are also shared memory islands.
    size_sm_sub = (size_sm-1) / 2 + 1;  // One may spilt also into more than 2 sub-islands
                                        // Rounding up with -1 / +1 trick
    color = my_rank_sm / size_sm_sub;
    MPI_Comm_split(comm_sm, color, 0, &comm_sm_sub);
  }
#elif defined(split_method_OPENMPI)
  // Of course, one can spilt MPI_COMM_WORLD directly into its NUMA domains.
  // Here we split comm_sm into its NUMA domains.
  MPI_Comm_split_type(comm_sm, OMPI_COMM_TYPE_NUMA, 0, MPI_INFO_NULL, &comm_sm_sub);
  // possible split types are: MPI_COMM_TYPE_SHARED,
  //   OMPI_COMM_TYPE_NODE, OMPI_COMM_TYPE_HWTHREAD, OMPI_COMM_TYPE_CORE, OMPI_COMM_TYPE_L1CACHE,
  //   OMPI_COMM_TYPE_L2CACHE, OMPI_COMM_TYPE_L3CACHE, OMPI_COMM_TYPE_SOCKET, OMPI_COMM_TYPE_NUMA,
  //   OMPI_COMM_TYPE_BOARD, OMPI_COMM_TYPE_HOST, OMPI_COMM_TYPE_CU, OMPI_COMM_TYPE_CLUSTER
#elif defined(split_method_MPICH)
  { MPI_Info info;
    MPI_Info_create (&info);
    MPI_Info_set(info, "SHMEM_INFO_KEY", "NUMA");  // This is not yet verified and tested :-(
    MPI_Comm_split_type(comm_sm, MPIX_COMM_TYPE_NEIGHBORHOOD, 0, info, &comm_sm_sub);
  }
#else
  // no further splitting
  comm_sm_sub = comm_sm;
#endif
  MPI_Comm_rank(comm_sm_sub, &my_rank_sm_sub);
  MPI_Comm_size(comm_sm_sub, &size_sm_sub);

  right = (my_rank_sm_sub+1)             % size_sm_sub;
  left  = (my_rank_sm_sub-1+size_sm_sub) % size_sm_sub;

  /* Allocate the window. */
  MPI_Win_allocate_shared((MPI_Aint) sizeof(int), sizeof(int), MPI_INFO_NULL, comm_sm_sub, &rcv_buf_ptr, &win);

  sum = 0;
  snd_buf = my_rank_sm_sub;

  for( i = 0; i < size_sm_sub; i++) 
  {
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPRECEDE, win);
    MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win);
    MPI_Win_fence(MPI_MODE_NOSTORE | MPI_MODE_NOPUT | MPI_MODE_NOSUCCEED, win);
    
    snd_buf = *rcv_buf_ptr;
    sum += *rcv_buf_ptr;
  }

  printf ("World: %i of %i comm_sm: %i of %i comm_sm_sub: %i of %i l/r=%i/%i Sum = %i\n", 
          my_rank_world,size_world, my_rank_sm,size_sm, 
          my_rank_sm_sub,size_sm_sub, left,right,  sum);

  MPI_Win_free(&win);

  MPI_Finalize();
}

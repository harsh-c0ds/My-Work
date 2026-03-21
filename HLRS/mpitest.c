/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the High Performance           *
 * Computing Centre Stuttgart (HLRS).                           *
 * It is made freely available with the understanding that      *
 * every copy of this file must include this header and that    *
 * HLRS take no responsibility for the use of the               *
 * enclosed teaching material.                                  *
 *                                                              *
 * Authors: Rolf Rabenseifner (HLRS)                            *
 *                                                              *
 * Contact: rabenseifner@hlrs.de                                * 
 *                                                              *  
 * Purpose: Check version of the MPI library and include file   *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_rank, size;
  int version, subversion;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Get_version(&version, &subversion);
 
  if (my_rank == 0)
  { if (size > 1)
    {
      printf ("Successful first MPI test executed in parallel on %i processes.\n", size);
    } else
    {
      printf ("Caution: This MPI test is executed only on one MPI process, i.e., sequentially!\n");
    }
    printf ("Your installation supports MPI standard version %i.%i.\n", version, subversion);
  }

  MPI_Finalize();
}

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
 * Purpose: A program to try MPI_Intercomm_create               *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/


#include <stdio.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
  int my_world_rank, world_size, my_sub_rank, sub_size, remote_leader, my_inter_rank;
  int sumA, sumB, sumC, sumD;
  int mycolor;
  MPI_Comm  sub_comm, inter_comm;

  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);

  mycolor = (my_world_rank > (world_size-1)/3);
  /* This definition of mycolor implies that the first color is 0
     --> see calculation of remote_leader below */
  MPI_Comm_split(MPI_COMM_WORLD, mycolor, 0, &sub_comm);
  MPI_Comm_size(sub_comm, &sub_size);
  MPI_Comm_rank(sub_comm, &my_sub_rank);

  /* Compute sum of all ranks. */
  MPI_Allreduce (&my_world_rank, &sumA, 1, MPI_INT, MPI_SUM, sub_comm); 
  MPI_Allreduce (&my_sub_rank,   &sumB, 1, MPI_INT, MPI_SUM, sub_comm); 

    /* local leader in the lower group is locally rank 0
         (to be provided in the lower group), 
       and within MPI_COMM_WORLD (i.e., in peer_comm) rank 0
         (to be provided in the upper group) */ 
    /* local leader in the upper group is locally rank 0
         (to be provided in the upper group), 
       and within MPI_COMM_WORLD (i.e., in peer_comm) 
       rank 0+(size of lower group)
         (to be provided in the lower group) */ 
  if (mycolor==0) /* This "if(...)" requires that mycolor==0 is the lower group! */
  { /*lower group*/
    remote_leader = 0 + sub_size;    
  }else{ /*upper group*/
    remote_leader = 0;
  }
  
  MPI_Intercomm_create(sub_comm,0,MPI_COMM_WORLD,remote_leader,0,&inter_comm);
  MPI_Comm_rank(inter_comm, &my_inter_rank);

  MPI_Allreduce(&my_inter_rank, &sumC, 1, MPI_INT, MPI_SUM, sub_comm); 
  MPI_Allreduce(&my_inter_rank, &sumD, 1, MPI_INT, MPI_SUM, inter_comm);

  printf ("PE world:%3i, color=%i sub:%3i inter:%3i SumA=%3i, SumB=%3i, SumC=%3i, SumD=%3i\n", 
          my_world_rank, mycolor, my_sub_rank, my_inter_rank, sumA, sumB, sumC, sumD);

  MPI_Finalize();
}

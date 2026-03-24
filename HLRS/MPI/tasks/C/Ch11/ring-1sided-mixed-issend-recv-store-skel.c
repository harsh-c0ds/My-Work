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
 * Purpose: The original pt-to-pt halo communication in a ring  *
 *          through all processes should be kept between the    *
 *          sub-islands and substituted with shared memory store*
 *          within the sub-islands.                             *
 *          Take care that the synchronization does not deadlock*
 *          even if the sub-islands contain only one process.   *
 *          Instead of the comm_sm shared memory islands, we    *
 *          use smaller sub-islands, because running on a       *
 *          shared system, one can still have more then one     *
 *          such sub-islands. In this exercise, we therefore    *
 *          communicate through pt-to-pt within MPI_COMM_WORLD  *
 *          or through shared memory assignments in comm_sm_sub.*
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
  int my_rank_sm_sub,size_sm_sub, color, left_sm_sub, right_sm_sub;
  MPI_Comm  comm_sm, comm_sm_sub;
  MPI_Group grp_world, grp_sm_sub; 
  int snd_buf;
  int *rcv_buf_ptr;
  int right, left;
  int sum, i;

  MPI_Status  status;
  MPI_Request request;

  MPI_Win     win;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank_world);
  MPI_Comm_size(MPI_COMM_WORLD, &size_world);

  // original calculation of the neighbors within MPI_COMM_WORLD
  right = (my_rank_world+1)            % size_world;
  left  = (my_rank_world-1+size_world) % size_world;

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &comm_sm);
  MPI_Comm_rank(comm_sm, &my_rank_sm);
  MPI_Comm_size(comm_sm, &size_sm);
  if (my_rank_world == 0)
  { if (size_sm == size_world) 
    {  printf("MPI_COMM_WORLD consists of only one shared memory region\n");
    }else
    { printf("MPI_COMM_WORLD is split into 2 or more shared memory islands\n");
  } }

  // Splitting comm_sm into smaller sub-islands. Of course, they are also shared memory islands.
  size_sm_sub = size_sm / 2;  // One may spilt also into more than 2 sub-islands
  if (size_sm_sub == 0) size_sm_sub = 1;
  color = my_rank_sm / size_sm_sub;
  MPI_Comm_split(comm_sm, color, 0, &comm_sm_sub);
  MPI_Comm_rank(comm_sm_sub, &my_rank_sm_sub);
  MPI_Comm_size(comm_sm_sub, &size_sm_sub);

  /* Allocate the window within the sub-islands. */
  MPI_Win_allocate_shared((MPI_Aint) sizeof(int), sizeof(int), MPI_INFO_NULL, comm_sm_sub, &rcv_buf_ptr, &win);

  // Is my neighbor in MPI_COMM_WORLD accessible within comm_sm_sub?
  MPI_Comm_group(MPI_COMM_WORLD, &grp_world);
  MPI_Comm_group(comm_sm_sub, &grp_sm_sub);

  // check for left neighbor: (for simplification, two calls are used instead of setting up an array of ranks)
  MPI_Group_translate_ranks(grp_world, 1, &left, grp_sm_sub, &left_sm_sub);
  // if left_sm_sub != MPI_UNDEFINED then receive from left is possible through comm_sm_sub

  // check for right neighbor:
  MPI_Group_translate_ranks(grp_world, 1, &right, grp_sm_sub, &right_sm_sub);
  // if right_sm_sub != MPI_UNDEFINED then send to right is possible through comm_sm_sub

  sum = 0;
  snd_buf = my_rank_world;

  for( i = 0; i < size_world; i++) 
  { // Please activate the //-lines and fill in the right choices for mixed communication.
    // Current code uses pt-to-pt for all communication

    //if(_____________________________)
        MPI_Issend(&snd_buf, 1, MPI_INT, right, 17, MPI_COMM_WORLD, &request);
    //if(_____________________________)
        MPI_Recv  (rcv_buf_ptr, 1, MPI_INT, left,  17, MPI_COMM_WORLD, &status);
    //if(_____________________________)
        MPI_Wait(&request, &status);

    //if(_____________________________) MPI_Win_fence(/*workaround: no assertions:*/ 0, win);
    // /* MPI_Put(&snd_buf, 1, MPI_INT, right, (MPI_Aint) 0, 1, MPI_INT, win); */
    // /*   ... is substituted by (with offset "right-my_rank" to store into right neighbor's rcv_buf): */
    //if(_____________________________)
    //       *(rcv_buf_ptr+(___________________________)) = snd_buf;
    //if(_____________________________) MPI_Win_fence(/*workaround: no assertions:*/ 0, win);
    
    snd_buf = *rcv_buf_ptr;
    sum += *rcv_buf_ptr;
  }

  printf ("World: %i of %i l/r=%i/%i comm_sm: %i of %i comm_sm_sub: %i of %i l/r=%i/%i Sum = %i\n", 
          my_rank_world,size_world, left,right, my_rank_sm,size_sm, 
          my_rank_sm_sub,size_sm_sub, left_sm_sub,right_sm_sub,  sum);

  MPI_Win_free(&win);

  MPI_Finalize();
}

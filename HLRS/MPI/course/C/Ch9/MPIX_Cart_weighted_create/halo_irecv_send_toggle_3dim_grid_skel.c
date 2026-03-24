/****************************************************************
 *                                                              *
 * This file has been written as a sample solution to an        *
 * exercise in a course given at the High Performance           *
 * Computing Centre Stuttgart (HLRS).                           *
 *                                                              *
 * Author:  Rolf Rabenseifner (HLRS)                            *
 *                                                              *
 * Purpose: A program to meassure 1-dim halo communication      *
 *          in myrank -1 and +1 directions (left and right)     *
 *                                                              *
 *          Test of MPIX_Cart_weighted_create() and ...         *
 *                                                              *
 * Toggle:  snd_buf and rcv_buf are exchanged after each        *
 *          communication step to prohibit wrong reporting      *
 *          of the CPU-to-_CPU communication.                   *
 *          This benchmark still reports sending out from       *
 *          warm caches (i.e., filled caches).                  *
 *                                                              *
 * Contents: C-Source                                           *
 *                                                              *
 ****************************************************************/

/** Proposal for improvements of the MPI API related to application aware
 * hardware topology mapping.
 *
 * @author Christoph Niethammer and Rolf Rabenseifner, 
 * High Performance Computing Center Stuttgart (HLRS), 
 * University of Stuttgart, Germany, 2018.
 *
 * Copyright (c) 2019, HLRS, University of Stuttgart, Germany.
 * All rights reserved.
 *
 * This software is made available under the BSD 3-clause license ("BSD License 2.0"):
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the copyright holder nor the names of its contributors
 *       may be used to endorse or promote products derived from this software 
 *       without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER NOR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>
#include "MPIX_interface_proposal.h"

#define number_of_messages 50
#define length_factor 2
#define number_package_sizes 10

#define ndims 3

int main(int argc, char *argv[])
{
  int i, j, length, my_world_rank, my_rank, size, test_value, mid, d;
  double start, finish, transfer_time; 
  MPI_Request rq[2];
  MPI_Status status_arr[2];
  MPI_Comm comm_cart, sub_comm[ndims];
  int dims[ndims], periods[ndims], remain_dims[ndims], coords[ndims], left[ndims], right[ndims];
  float *snd_buf_left, *snd_buf_right;
  float *rcv_buf_left, *rcv_buf_right;

  int meshsize_avg_per_proc_startval[ndims];
  int meshsize_avg_per_proc[ndims], meshsize_local[ndims], halosize[ndims], real_halosize[ndims], max_buffersize, total_halosize;
  long long meshsize_total;
  int blocklength[ndims], stride[ndims], cnt[ndims];
  double weights[ndims];
  MPI_Datatype datatype[ndims];

  int cart_method=99;
  /* 0 = end
     1 = MPI_Dims_create + MPI_Cart_create
     2 = MPIX_Cart_weighted_create ( MPIX_WEIGHTS_EQUAL )
     3 = MPIX_Cart_weighted_create ( weights )
     4 = MPIX_Cart_ml_create ( dims_ml )
    99 = start value (used only for first while iteration)
  */

/* Naming conventions                                                                */
/* Processes:                                                                        */
/*     my_rank-1                        my_rank                         my_rank+1    */
/* "left neighbor"                     "myself"                     "right neighbor" */
/*   ...    rcv_buf_right <--- snd_buf_left snd_buf_right ---> rcv_buf_left    ...   */
/*   ... snd_buf_right ---> rcv_buf_left       rcv_buf_right <--- snd_buf_left ...   */
/*                        |                                  |                       */
/*              halo-communication                 halo-communication                */

 MPI_Init(&argc, &argv);
 MPI_Comm_size(MPI_COMM_WORLD, &size);
 MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);

while (cart_method > 0) {
 MPI_Barrier(MPI_COMM_WORLD); /* that previous cart_method is completely finished */
 if (my_world_rank==0) {
   printf("\n\n[I/O via stdout AND stdin through rank==0 in MPI_COMM_WORLD]\n");
   printf("\nGeneral input questions:\n\n");
   printf("cart_method: 0=end, 1=Dims_create+Cart_create, 2=Cart_weighted_create(WEIGHTS_EQUAL), 3=dito(weights), 4=dito manually, 5=Cart_ml_create(dims_ml) ?\n");
   scanf("%d",&cart_method);
 }
 MPI_Bcast(&cart_method, 1 ,MPI_INT, 0, MPI_COMM_WORLD);
 if (cart_method > 0) 
 { 
  if (my_world_rank==0) {
    printf("start mesh sizes integer start values for %d dimensions (e.g., 2 2 2) ?\n", ndims);
    for (d=0; d<ndims; d++) scanf("%d",&meshsize_avg_per_proc_startval[d]);
  }
  MPI_Bcast(meshsize_avg_per_proc_startval, ndims, MPI_INT, 0, MPI_COMM_WORLD);

  if (my_world_rank==0) {
    printf("blocklength and stride of MPI_TYPE_VECTOR for each of the %d dimensions, '0 0' stands for contiguous (e.g. 0 0 256 1024 8 32) ?\n", ndims);
    for (d=0; d<ndims; d++) { scanf("%d",&blocklength[d]); scanf("%d",&stride[d]); }
  }
  MPI_Bcast(blocklength, ndims, MPI_INT, 0, MPI_COMM_WORLD); MPI_Bcast(stride, ndims, MPI_INT, 0, MPI_COMM_WORLD);

  for (d=0; d<ndims; d++) {
    if (blocklength[d]==0 || stride[d]==0) { blocklength[d] = 1;  stride[d]=0; }
    periods[d]=1;
    dims[d]=0;
  } 

  if (my_world_rank==0) {
    printf("\nReporting of all general input arguments:\n\n");
    printf("cart_method = %d\n", cart_method);
    printf("start mesh sizes integer start values for %d dimensions =", ndims);
    for (d=0; d<ndims; d++) printf(" %d",meshsize_avg_per_proc_startval[d]); printf("\n");
    printf("blocklength & sgtride pairs for each of the %d dimensions =", ndims);
    for (d=0; d<ndims; d++) printf(" %d %d  ", blocklength[d], stride[d]); printf("\n");
    printf("\nCreating the Cartesian communicator and further input arguments:\n\n");
  } 
  
  if (cart_method == 1) {
     if (my_world_rank==0) printf("cart_method == 1: MPI_Dims_create + MPI_Cart_create\n");
     MPI_Dims_create(size, ndims, dims);
     MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 0, &comm_cart); 
  } else if (cart_method == 2) {
     if (my_world_rank==0) printf("cart_method == 2: MPIX_Cart_weighted_create( MPIX_WEIGHTS_EQUAL )\n");

     /* TODO: Appropriate call to MPIX_Cart_weighted_create(...) with MPIX_WEIGHTS_EQUAL 
              instead of calling MPI_Dims_create() and MPI_Cart_create() as in method 1 */

  } else if (cart_method == 3) {  
     
     /* TODO: Appropriate calculation of weights[] based on meshsize_avg_per_proc_startval[] */
     
     if (my_world_rank==0) { printf("cart_method == 3: MPIX_Cart_weighted_create( weights := ___________TODO_____________________ )\n");
       printf("weights= "); for (d=0; d<ndims; d++) printf(" %lf",weights[d]); printf("\n");
     }

     /* TODO: Appropriate call to MPIX_Cart_weighted_create(...) with weights 
              instead of MPIX_WEIGHTS_EQUAL as in method 2 */

  } else if (cart_method == 4) {
     for (d=0; d<ndims; d++) weights[d] = 4.0 / meshsize_avg_per_proc_startval[d];
     if (my_world_rank==0) { printf("cart_method == 4: MPIX_Cart_weighted_create( manual weights )\n");
       printf("weights (double values) for %d dimensions (e.g., ", ndims);
       for (d=0; d<ndims; d++) printf(" %lf",weights[d]); printf(") ?\n");
       for (d=0; d<ndims; d++) scanf("%lf",&weights[d]);
       printf("weights= "); for (d=0; d<ndims; d++) printf(" %lf",weights[d]); printf("\n");
     }
     MPI_Bcast(weights, ndims, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     MPIX_Cart_weighted_create(MPI_COMM_WORLD, ndims, weights, periods, MPI_INFO_NULL, dims, &comm_cart);
  } else { /*cart_method==5*/
    int nlevels, level;
    if (my_world_rank==0) { printf("number of hardware levels (e.g., 3, must be at least 1) ?\n");
      scanf("%d",&nlevels);  
    }
    MPI_Bcast(&nlevels, 1, MPI_INT, 0, MPI_COMM_WORLD);
    { int dims_ml[ndims][nlevels];
      if (my_world_rank==0) { 
        printf("Choose dims_ml: for each of the %d Cartesian dimensions a list of %d dimensions from outer to inner hardware level ?\n", ndims, nlevels);
        printf("The product of all values must be the size of MPI_COMM_WORLD = %d\n", size);
        for (d=0; d<ndims; d++) for (level=0; level<nlevels; level++) scanf("%d", &dims_ml[d][level]);
        printf("nlevels = %d\ndims_ml = ", nlevels);
        for (d=0; d<ndims; d++) for (level=0; level<nlevels; level++) printf(" %d", dims_ml[d][level]);
      }
      MPI_Bcast(dims_ml, ndims*nlevels, MPI_INT, 0, MPI_COMM_WORLD);
      MPIX_Cart_ml_create(MPI_COMM_WORLD, ndims, periods, nlevels, dims_ml, MPI_INFO_NULL, dims, &comm_cart);
    }
  }
  MPI_Comm_rank(comm_cart, &my_rank);
  MPI_Cart_coords(comm_cart, my_rank, ndims, coords);
  
 /*
  meshsize_total = 1; max_buffersize=0;
  for (d=0; d<ndims; d++)
  { meshsize_avg_per_proc[d] = meshsize_avg_per_proc_startval[d];
    for (j = 1; j < number_package_sizes; j++) 
       meshsize_avg_per_proc[d] = meshsize_avg_per_proc[d] * length_factor;
    meshsize_local[d] = (int)(meshsize_avg_per_proc[d] 
                            * pow(1.0*size, 1.0/ndims)) - 1) / dims[d] + 1;
    meshsize_total = meshsize_total * meshsize_local[d];
  }
  for (d=0; d<ndims; d++)
  { halosize[d] = meshsize_total / meshsize_local[d]; 
    if (halosize[d] > max_buffersize) max_buffersize = halosize[d];
  } 
 */

  MPI_Barrier(MPI_COMM_WORLD); /* Because we switch I/O from rank 0 in MPI_COMM_WORLD (stdout+stdin) to rank 0 in comm_cart (only stdout) */
  sleep(1);                    /* We expect that all I/O above by rank 0 in MPI_COMM_WORLD is finished within 1 second */
  MPI_Barrier(comm_cart);      /* For the case that rank 0 in MPI_COMM_WORLD left the barrier on MPI_COMM_WORLD earlier than the rank 0 in comm_cart
                                  this barrier should guarantee that rank 0 in comm_cart starts at least 1 second after rank 0 in MPI_COMM_WORLD
                                  left its barrier on MPI_COMM_WORLD */
  if (my_rank == 0) 
  { 
    printf("\n[MPI_Barrier and switching to output via stdout through rank==0 in comm_cart]\n\n");
    printf("ndims=%d dims=", ndims); for (d=0; d<ndims; d++) printf(" %d", dims[d]); printf("\n");
   /*
    printf("  max avg meshsize/proc="); for (d=0; d<ndims; d++) printf(" %d", meshsize_avg_per_proc[d]); printf("\n");
    printf("  max local meshsize   ="); for (d=0; d<ndims; d++) printf(" %d", meshsize_local[d]); printf("\n");
    printf("  max total meshsize   = %lld\n", meshsize_total);
    printf("  max halosize         ="); for (d=0; d<ndims; d++) printf(" %d", halosize[d]); printf("\n");
    printf("  max of max halosize  = %d\n", max_buffersize);
   */
  }

  for (d=0; d<ndims; d++)
  {
    for (i=0; i<ndims; i++) remain_dims[i]=0;
    remain_dims[d]=1;
    MPI_Cart_sub(comm_cart, remain_dims, &sub_comm[d]);
    MPI_Cart_shift(sub_comm[d], 0, 1, &left[d], &right[d]);
    /* or MPI_Cart_shift(cart_comm, d, 1, &left[d], &right[d]); */
    /* or right[d] = (coords[d]+1)         % dims[d]; */
    /*    left[d]  = (coords[d]-1+dims[d]) % dims[d]; */
  }

  MPI_Barrier(comm_cart); /* that all processes start the meassurement round together */
  if (my_rank == 0) printf("    message size      transfertime  duplex bandwidth per process and neighbor (mesh&halo in #floats)\n");

  for (d=0; d<ndims; d++) meshsize_avg_per_proc[d] = meshsize_avg_per_proc_startval[d];

  for (j = 1; j <= number_package_sizes; j++)
  { 
    total_halosize = 0;    /* Caution: the real total message size is 2*total_halosize, due to sending left and right */
    max_buffersize =0;
    meshsize_total = 1;
    for (d=0; d<ndims; d++)
    { meshsize_local[d] = ((int)(meshsize_avg_per_proc[d] * pow(1.0*size, 1.0/ndims)) - 1) / dims[d] + 1;
      meshsize_total = meshsize_total * meshsize_local[d];
    }
    for (d=0; d<ndims; d++) { 
      int buffersize; 
      halosize[d] = meshsize_total / meshsize_local[d]; 
      if (stride[d] == 0) {
        datatype[d] = MPI_FLOAT;  cnt[d] = halosize[d];  real_halosize[d] = halosize[d];  buffersize = halosize[d];
      } else { 
        int blocks;
        blocks = (halosize[d]-1)/blocklength[d]+1;
        if (blocks < 1) blocks = 1;
        MPI_Type_vector(blocks, blocklength[d], stride[d], MPI_FLOAT, &datatype[d]);
        MPI_Type_commit(&datatype[d]);
        /* if (my_rank==0) printf("d=%d: MPI_Type_vector(blocks=%d, blocklength[d]=%d, stride[d]=%d,...)\n", d, blocks, blocklength[d], stride[d]); */
        cnt[d] = 1;
        real_halosize[d] = blocks * blocklength[d];
        buffersize = blocks * stride[d];
      }
      total_halosize += real_halosize[d];
      if (buffersize > max_buffersize) max_buffersize = buffersize;
    }

    snd_buf_left  = malloc(max_buffersize*sizeof(float));
    snd_buf_right = malloc(max_buffersize*sizeof(float));
    rcv_buf_left  = malloc(max_buffersize*sizeof(float));
    rcv_buf_right = malloc(max_buffersize*sizeof(float));
    
    for (i = 0; i <= number_of_messages; i++)
    {
      if(i==1) start = MPI_Wtime();

     for (d=0; d<ndims; d++)
     {
      length = cnt[d];
      test_value = j*1000000 + i*10000 + coords[d]*10 ; mid = (length-1)/number_of_messages*i;

      snd_buf_left[0]=test_value+1  ; snd_buf_left[mid]=test_value+2  ; snd_buf_left[length-1]=test_value+3;
      snd_buf_right[0]=test_value+6 ; snd_buf_right[mid]=test_value+7 ; snd_buf_right[length-1]=test_value+8;

      MPI_Irecv(rcv_buf_right, length, datatype[d], right[d], 17+d, sub_comm[d], &rq[0]);
      MPI_Irecv(rcv_buf_left,  length, datatype[d], left[d],  23+d, sub_comm[d], &rq[1]);

      MPI_Send(snd_buf_left,  length, datatype[d], left[d],  17+d, sub_comm[d]);
      MPI_Send(snd_buf_right, length, datatype[d], right[d], 23+d, sub_comm[d]);
    
      MPI_Waitall(2, rq, status_arr);

/*    ...snd_buf_... is used to store the values that were stored in snd_buf_... in the neighbor process */
      test_value = j*1000000 + i*10000 + left[d]*10  ; mid = (length-1)/number_of_messages*i;
      snd_buf_right[0]=test_value+6 ; snd_buf_right[mid]=test_value+7 ; snd_buf_right[length-1]=test_value+8;
      test_value = j*1000000 + i*10000 + right[d]*10 ; mid = (length-1)/number_of_messages*i;
      snd_buf_left[0]=test_value+1  ; snd_buf_left[mid]=test_value+2  ; snd_buf_left[length-1]=test_value+3;
      if ((rcv_buf_left[0] != snd_buf_right[0]) || (rcv_buf_left[mid] != snd_buf_right[mid]) || 
                                                   (rcv_buf_left[length-1] != snd_buf_right[length-1])) {
         printf("%d: d=%d, j=%d, i=%d --> snd_buf_right[0,%d,%d]=(%f,%f,%f)\n",
                    my_rank, d, j, i, mid, length-1, snd_buf_right[0], snd_buf_right[mid], snd_buf_right[length-1]);
         printf("%d:     is not identical to rcv_buf_left[0,%d,%d]=(%f,%f,%f)\n",
                    my_rank,       mid, length-1, rcv_buf_left[0],  rcv_buf_left[mid],  rcv_buf_left[length-1]);
      }
      if ((rcv_buf_right[0] != snd_buf_left[0]) || (rcv_buf_right[mid] != snd_buf_left[mid]) ||
                                                   (rcv_buf_right[length-1] != snd_buf_left[length-1])) {
         printf("%d: d=%d, j=%d, i=%d --> snd_buf_left[0,%d,%d]=(%f,%f,%f)\n",
                    my_rank, d, j, i, mid, length-1, snd_buf_left[0],  snd_buf_left[mid],  snd_buf_left[length-1]);
         printf("%d:     is not identical to rcv_buf_right[0,%d,%d]=(%f,%f,%f)\n",
                    my_rank,       mid, length-1, rcv_buf_right[0], rcv_buf_right[mid], rcv_buf_right[length-1]);
      }

      { /*toggle:*/ float *tmp;
        tmp=snd_buf_left;  snd_buf_left  = rcv_buf_right; rcv_buf_right = tmp;
        tmp=snd_buf_right; snd_buf_right = rcv_buf_left;  rcv_buf_left  = tmp;
      }
     }

    }
    finish = MPI_Wtime();

    if (my_rank == 0) 
    {
      transfer_time = (finish - start) / number_of_messages;
      printf("%10i bytes %12.3f usec %10.3f MB/s", /* 2*total_halosize due to sending left and right */
             2*total_halosize*(int)sizeof(float), transfer_time*1e6, 1.0e-6*2*total_halosize*sizeof(float) / transfer_time);
      printf(" meshsizes"); 
      printf(" total="); for (d=0; d<ndims; d++) printf(" %5d", meshsize_local[d]*dims[d]);
      printf(" per process="); for (d=0; d<ndims; d++) printf(" %4d", meshsize_local[d]);
      printf(" halosize= %8d =", total_halosize);
      for (d=0; d<ndims; d++) {
        printf(" + %d", real_halosize[d]); 
        if (stride[d]!=0) printf(" (B%d/S%d)", blocklength[d], stride[d]);
      }
      printf("\n"); 
    }

    for (d=0; d<ndims; d++) if (datatype[d] != MPI_FLOAT) MPI_Type_free(&datatype[d]);

    free(snd_buf_left );
    free(snd_buf_right);
    free(rcv_buf_left );
    free(rcv_buf_right);

    for (d=0; d<ndims; d++) meshsize_avg_per_proc[d] = meshsize_avg_per_proc[d] * length_factor;

  } /* end of for (j = 1; j <= number_package_sizes; j++) */

  for (d=0; d<ndims; d++) MPI_Comm_free(&sub_comm[d]);
  MPI_Comm_free(&comm_cart);

 } /* end of if (cart_method > 0) */
} /* end of while (cart_method > 0) */
 MPI_Finalize();
}

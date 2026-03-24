PROGRAM ring

!==============================================================!
!                                                              !
! This file has been written as a sample solution to an        !
! exercise in a course given at the High Performance           !
! Computing Centre Stuttgart (HLRS).                           !
! The examples are based on the examples in the MPI course of  !
! the Edinburgh Parallel Computing Centre (EPCC).              !
! It is made freely available with the understanding that      !
! every copy of this file must include this header and that    !
! HLRS and EPCC take no responsibility for the use of the      !
! enclosed teaching material.                                  !
!                                                              !
! Authors: Joel Malard, Alan Simpson,            (EPCC)        !
!          Rolf Rabenseifner, Traugott Streicher (HLRS)        !
!                                                              !
! Contact: rabenseifner@hlrs.de                                !
!                                                              !
! Purpose: A program to try out one-sided communication        !
!          with window=rcv_buf and MPI_PUT to put              !
!          local snd_buf value into remote window (rcv_buf).   !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : C_PTR, C_F_POINTER
  IMPLICIT NONE

  INTEGER :: my_rank_world, size_world
  INTEGER :: my_rank_sm,    size_sm
  INTEGER :: my_rank_sm_sub,size_sm_sub, color;
  TYPE(MPI_Comm) :: comm_sm, comm_sm_sub
  INTEGER :: right, left
  INTEGER :: i, sum
  INTEGER, ASYNCHRONOUS :: snd_buf
  INTEGER, POINTER, ASYNCHRONOUS :: rcv_buf(:)  ! "(:)" because it is an array
  TYPE(C_PTR) :: ptr_rcv_buf
  TYPE(MPI_Win) :: win 
  INTEGER :: disp_unit
  INTEGER(KIND=MPI_ADDRESS_KIND) :: integer_size, lb
  INTEGER(KIND=MPI_ADDRESS_KIND) :: rcv_buf_size, target_disp

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank_world)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size_world)

  CALL MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm_sm)
  CALL MPI_Comm_rank(comm_sm, my_rank_sm) 
  CALL MPI_Comm_size(comm_sm, size_sm) 
  IF (my_rank_world == 0) THEN
    IF (size_sm == size_world) THEN
      write (*,*) 'MPI_COMM_WORLD consists of only one shared memory region'
    ELSE
      write (*,*) 'MPI_COMM_WORLD is split into 2 or more shared memory islands'
    END IF
  END IF

! #define split_method_SPLIT
! // #define split_method_OPENMPI
! // #define split_method_MPICH
! #if defined(split_method_SPLIT)
    ! Splitting comm_sm into smaller sub-islands. Of course, they are also shared memory islands.
    _________________________________    ! One may spilt also into more than 2 sub-islands
                                         ! Rounding up with -1 / +1 trick
    _________________________________
    CALL MPI___________(comm_sm, ________, comm_sm_sub)
! #elif defined(split_method_OPENMPI)
!   // Of course, one can spilt MPI_COMM_WORLD directly into its NUMA domains.
!   // Here we split comm_sm into its NUMA domains.
!   CALL MPI_Comm_split_type(comm_sm, OMPI_COMM_TYPE_NUMA, 0, MPI_INFO_NULL, comm_sm_sub)
!   // possible split types are: MPI_COMM_TYPE_SHARED,
!   //   OMPI_COMM_TYPE_NODE, OMPI_COMM_TYPE_HWTHREAD, OMPI_COMM_TYPE_CORE, OMPI_COMM_TYPE_L1CACHE,
!   //   OMPI_COMM_TYPE_L2CACHE, OMPI_COMM_TYPE_L3CACHE, OMPI_COMM_TYPE_SOCKET, OMPI_COMM_TYPE_NUMA,
!   //   OMPI_COMM_TYPE_BOARD, OMPI_COMM_TYPE_HOST, OMPI_COMM_TYPE_CU, OMPI_COMM_TYPE_CLUSTER
! #elif defined(split_method_MPICH)
!   { MPI_Info info;
!     MPI_Info_create (&info);
!     MPI_Info_set(info, "SHMEM_INFO_KEY", "NUMA");  // This is not yet verified and tested :-(
!     MPI_Comm_split_type(comm_sm, MPIX_COMM_TYPE_NEIGHBORHOOD, 0, info, &comm_sm_sub);
!   }
! #else
!   // no further splitting
!   comm_sm_sub = comm_sm
! #endif
  CALL MPI_Comm_rank(comm_sm_sub, my_rank_sm_sub) 
  CALL MPI_Comm_size(comm_sm_sub, size_sm_sub) 

  right = mod(my_rank_sm_sub+1,         size_sm_sub)
  left  = mod(my_rank_sm_sub-1+size_sm_sub, size_sm_sub)

! ALLOCATE THE WINDOW.
  CALL MPI_Type_get_extent(MPI_INTEGER, lb, integer_size)
  rcv_buf_size = 1 * integer_size
  disp_unit = integer_size
  CALL MPI_Win_allocate_shared(rcv_buf_size, disp_unit, MPI_INFO_NULL, comm_sm_sub, ptr_rcv_buf, win)
  CALL C_F_POINTER(ptr_rcv_buf, rcv_buf, (/1/)) ! if rcv_buf is an array
  rcv_buf(0:) => rcv_buf ! change lower bound to 0 (instead of default 1) ! if rcv_buf is an array
! CALL C_F_POINTER(ptr_rcv_buf, rcv_buf) ! if rcv_buf is not an array

  sum = 0
  snd_buf = my_rank_sm_sub

  DO i = 1, size_sm_sub

!    ... The compiler may move the read access to rcv_buf
!        in the previous loop iteration after the following 
!        1-sided MPI calls, because the compiler has no chance
!        to see, that rcv_buf will be modified by the following 
!        1-sided MPI calls.  Therefore a dummy routine must be 
!        called with rcv_buf as argument:
 
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)
 
!    ... Now, the compiler expects that rcv_buf was modified,
!        because the compiler cannot see that MPI_F_SYNC_REG
!        did nothing. Therefore the compiler cannot move any
!        access to rcv_buf across this dummy call. 
 
     CALL MPI_Win_fence(MPI_MODE_NOSTORE + MPI_MODE_NOPRECEDE, win)

     target_disp = 0
     CALL MPI_Put(snd_buf, 1, MPI_INTEGER, right, target_disp, 1, MPI_INTEGER, win)

     CALL MPI_Win_fence(MPI_MODE_NOSTORE + MPI_MODE_NOPUT + MPI_MODE_NOSUCCEED, win)
 
!    ... The compiler has no chance to see, that rcv_buf was
!        modified. Therefore a dummy routine must be called
!        with rcv_buf as argument:
 
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)
 
!    ... Now, the compiler expects that rcv_buf was modified,
!        because the compiler cannot see that MPI_F_SYNC_REG
!        did nothing. Therefore the compiler will use the new
!        value on the memory, instead of some old value in a
!        register.

     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)
!    ... This dummy call with snd_buf in the argument list prevents
!        the following store access to snd_buf may be moved by the
!        compiler to a position between PUT(snd_buf...) and FENCE.

     snd_buf = rcv_buf(0)
     sum = sum + rcv_buf(0)

  END DO

  WRITE(*,*) 'World:',  my_rank_world,' of ',size_world, &
   &         'comm_sm:',my_rank_sm,   ' of ',size_sm, &
   &         'comm_sm_sub:',my_rank_sm_sub,   ' of ',size_sm_sub, &
   &                'l/r=',left,'/',right, &
   &         '; Sum =', sum

  CALL MPI_Win_free(win)

  CALL MPI_Finalize()

END PROGRAM

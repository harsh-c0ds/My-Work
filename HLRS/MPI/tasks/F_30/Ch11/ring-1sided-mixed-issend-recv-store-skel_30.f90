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
! Purpose: The original pt-to-pt halo communication in a ring  !
!          through all processes should be kept between the    !
!          sub-islands and substituted with shared memory store!
!          within the sub-islands.                             !
!          Take care that the synchronization does not deadlock!
!          even if the sub-islands contain only one process.   !
!          Instead of the comm_sm shared memory islands, we    !
!          use smaller sub-islands, because running on a       !
!          shared system, one can still have more then one     !
!          such sub-islands. In this exercise, we therefore    !
!          communicate through pt-to-pt within MPI_COMM_WORLD  !
!          or through shared memory assignments in comm_sm_sub.!
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08
  USE, INTRINSIC :: ISO_C_BINDING, ONLY : C_PTR, C_F_POINTER
  IMPLICIT NONE

  INTEGER :: my_rank_world, size_world
  INTEGER :: my_rank_sm,    size_sm
  INTEGER :: my_rank_sm_sub,size_sm_sub, color, left_sm_sub(1), right_sm_sub(1)
  TYPE(MPI_Comm)  :: comm_sm, comm_sm_sub
  TYPE(MPI_Group) :: grp_world, grp_sm_sub
  INTEGER :: right, left
  INTEGER :: i, sum
  INTEGER :: snd_buf  ! no longer ASYNCHRONOUS, because no MPI_Put(snd_buf, ...)
  INTEGER, POINTER, ASYNCHRONOUS :: rcv_buf(:)  ! "(:)" because it is an array
  TYPE(C_PTR) :: ptr_rcv_buf
  TYPE(MPI_Win) :: win 
  INTEGER :: disp_unit
  INTEGER(KIND=MPI_ADDRESS_KIND) :: integer_size, lb
  INTEGER(KIND=MPI_ADDRESS_KIND) :: rcv_buf_size, target_disp

  TYPE(MPI_Status)  :: status
  TYPE(MPI_Request) :: request

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank_world)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size_world)

! original calculation of the neighbors within MPI_COMM_WORLD
  right = mod(my_rank_world+1,            size_world)
  left  = mod(my_rank_world-1+size_world, size_world)

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

! Splitting comm_sm into smaller sub-islands. Of course, they are also shared memory islands.
  size_sm_sub = size_sm / 2  ! One may spilt also into more than 2 sub-islands
  if (size_sm_sub == 0) size_sm_sub = 1
  color = my_rank_sm / size_sm_sub
  CALL MPI_Comm_split(comm_sm, color, 0, comm_sm_sub)
  CALL MPI_Comm_rank(comm_sm_sub, my_rank_sm_sub)
  CALL MPI_Comm_size(comm_sm_sub, size_sm_sub)

! ALLOCATE THE WINDOW.
  CALL MPI_Type_get_extent(MPI_INTEGER, lb, integer_size)
  rcv_buf_size = 1 * integer_size
  disp_unit = integer_size
  CALL MPI_Win_allocate_shared(rcv_buf_size, disp_unit, MPI_INFO_NULL, comm_sm_sub, ptr_rcv_buf, win)
  CALL C_F_POINTER(ptr_rcv_buf, rcv_buf, (/1/)) ! if rcv_buf is an array
  rcv_buf(0:) => rcv_buf ! change lower bound to 0 (instead of default 1) ! if rcv_buf is an array
! CALL C_F_POINTER(ptr_rcv_buf, rcv_buf) ! if rcv_buf is not an array

! Is my neighbor in MPI_COMM_WORLD accessible within comm_sm_sub?
  CALL MPI_Comm_group(MPI_COMM_WORLD, grp_world)
  CALL MPI_Comm_group(comm_sm_sub, grp_sm_sub)

! check for left neighbor: (for simplification, two calls are used instead of setting up an array of ranks)
  CALL MPI_Group_translate_ranks(grp_world, 1, (/ left /), grp_sm_sub, left_sm_sub)
! if left_sm_sub(1) /= MPI_UNDEFINED then receive from left is possible through comm_sm_sub

! check for right neighbor:
  CALL MPI_Group_translate_ranks(grp_world, 1, (/ right /), grp_sm_sub, right_sm_sub)
! if right_sm_sub(1) /= MPI_UNDEFINED then send to right is possible through comm_sm_sub

  sum = 0
  snd_buf = my_rank_world

  DO i = 1, size_world
    !  Please activate the !!-lines and fill in the right choices for mixed communication.
    !  Current code uses pt-to-pt for all communication.

    !! IF (________________________________) THEN
       CALL MPI_Issend(snd_buf, 1, MPI_INTEGER, right, 17, MPI_COMM_WORLD, request)
    !! ENDIF
    !! IF (________________________________) THEN
       CALL MPI_Recv  (rcv_buf, 1, MPI_INTEGER, left,  17, MPI_COMM_WORLD, status)
    !! ENDIF
    !! IF (________________________________) THEN
       CALL MPI_Wait(request, status)
    !! ENDIF
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)

!    ... The compiler may move the read access to rcv_buf
!        in the previous loop iteration after the following 
!        1-sided MPI calls, because the compiler has no chance
!        to see, that rcv_buf will be modified by the following 
!        1-sided MPI calls.  Therefore a dummy routine must be 
!        called with rcv_buf as argument:
    !! IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)
!    ... Now, the compiler expects that rcv_buf was modified,
!        because the compiler cannot see that MPI_F_SYNC_REG
!        did nothing. Therefore the compiler cannot move any
!        access to rcv_buf across this dummy call. 
 
    !! IF (________________________________) THEN
    !!  CALL MPI_WIN_FENCE(0, win)  ! Workaround: no assertions
    !! ENDIF
 
    !! IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)
!    ... that the "rcv_buf(0+(right-my_rank_sm)) = snd_buf" statement
!        cannot be moved across then "CALL MPI_Win_fence" above  

    !! IF (________________________________) THEN
!      target_disp = 0
!      CALL MPI_Put(snd_buf, 1, MPI_INTEGER, right, target_disp, 1, MPI_INTEGER, win)
    !!   rcv_buf(0+(________________________________)) = snd_buf
    !! ENDIF
 
    !! IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)
!    ... that the "rcv_buf(0+(right-my_rank_sm)) = snd_buf" statement
!        cannot be moved across then "CALL MPI_Win_fence" below

    !! IF (________________________________) THEN
    !!  CALL MPI_WIN_FENCE(0, win)  ! Workaround: no assertions
    !! ENDIF
 
!    ... The compiler has no chance to see, that rcv_buf was
!        modified. Therefore a dummy routine must be called
!        with rcv_buf as argument:
    !! IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)
!    ... Now, the compiler expects that rcv_buf was modified,
!        because the compiler cannot see that MPI_F_SYNC_REG
!        did nothing. Therefore the compiler will use the new
!        value on the memory, instead of some old value in a
!        register.

     snd_buf = rcv_buf(0)
     sum = sum + rcv_buf(0)

  END DO

  WRITE(*,*) 'World:',  my_rank_world,' of ',size_world, &
   &                'l/r=',left,'/',right, &
   &         'comm_sm:',my_rank_sm,   ' of ',size_sm, &
   &         'comm_sm_sub:',my_rank_sm_sub,   ' of ',size_sm_sub, &
   &                'l/r=',left_sm_sub(1),'/',right_sm_sub(1), &
   &         '; Sum =', sum

  CALL MPI_Win_free(win)

  CALL MPI_Finalize()

END PROGRAM

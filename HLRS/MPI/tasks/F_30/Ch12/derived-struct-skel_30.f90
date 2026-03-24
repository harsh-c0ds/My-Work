PROGRAM ring_derived

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
! Purpose: A program that uses derived data-types.             !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08
  IMPLICIT NONE

  INTEGER :: my_rank, size
  INTEGER :: i, right, left

  TYPE t
     SEQUENCE
     INTEGER :: i
     REAL    :: r
  END TYPE t
  TYPE(t), ASYNCHRONOUS :: snd_buf
  TYPE(t) :: rcv_buf, sum

  INTEGER(KIND=MPI_ADDRESS_KIND) :: first_var_address, second_var_address
  TYPE(MPI_Datatype) :: send_recv_type

  INTEGER :: array_of_block_length(2)
  TYPE(MPI_Datatype) :: array_of_types(2)
  INTEGER(KIND=MPI_ADDRESS_KIND) :: array_of_displacements(2)

  TYPE(MPI_Status) :: status
  TYPE(MPI_Request) :: request

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)
  right = mod(my_rank+1,      size)
  left  = mod(my_rank-1+size, size)

! Create derived datatype. 

  array_of_block_length(1) = ___    ! PLEASE SUBSTITUTE ALL SKELETON CODE: ____ 
  array_of_block_length(2) = ___

  array_of_types(1) = ___
  array_of_types(2) = ___

  CALL MPI_Get_address(snd_buf%i, first_var_address)
  CALL MPI_Get_address(snd_buf%r, second_var_address)

  array_of_displacements(1) = 0
  array_of_displacements(2) = ___

  CALL MPI_Type_create_struct(___ ... ___, send_recv_type)
  CALL MPI____________(send_recv_type)

! ---------- original source code from MPI/course/F_30/Ch4/ring_30.f90 - PLEASE MODIFY: 
  sum%i = 0
  sum%r = 0
  snd_buf%i = my_rank           ! Step 1
  snd_buf%r = REAL(10*my_rank)  ! Step 1

  DO i = 1, size
     CALL MPI_Issend(snd_buf, 1, MPI_INTEGER, right, 17, MPI_COMM_WORLD, request)    ! Step 2a
     CALL MPI_Recv(rcv_buf, 1, MPI_INTEGER, left, 17, MPI_COMM_WORLD, status)        ! Step 3
     CALL MPI_Wait(request, status)                                                  ! Step 2b
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)           ! Step 2c
     snd_buf = rcv_buf                                                               ! Step 4
     sum%i = sum%i + rcv_buf%i                                                       ! Step 5
     sum%r = sum%r + rcv_buf%r                                                       ! Step 5
  END DO

  WRITE(*,*) "PE", my_rank, ": Sum%i =", sum%i, " Sum%r =", sum%r

  CALL MPI_Finalize()

END PROGRAM

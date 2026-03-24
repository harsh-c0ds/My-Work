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
  IMPLICIT NONE

  INTEGER :: my_rank, size
  INTEGER :: right, left
  INTEGER :: i, sum
  INTEGER, ASYNCHRONOUS :: snd_buf
  INTEGER :: rcv_buf
  TYPE(MPI_Status)  :: status
  TYPE(MPI_Request) :: request
  _____________ :: win 
  _____________ :: disp_unit
  _____________ :: integer_size, lb
  _____________ :: rcv_buf_size, target_disp

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)

  right = mod(my_rank+1,      size)
  left  = mod(my_rank-1+size, size)

! CREATE THE WINDOW.
  CALL MPI_Type_get_extent(MPI_INTEGER, lb, integer_size)
  rcv_buf_size = _______________
  disp_unit = __________________
  CALL MPI_Win_create( _________________ ,win) 

  sum = 0
  snd_buf = my_rank

  DO i = 1, size

! CAUTION: This source file still contains the old nonblocking pt-to-pt communication
     CALL MPI_Issend(snd_buf, 1, MPI_INTEGER, right, 17, MPI_COMM_WORLD, request)
     CALL MPI_Recv  (rcv_buf, 1, MPI_INTEGER, left,  17, MPI_COMM_WORLD, status)
     CALL MPI_Wait(request, status)
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)

     snd_buf = rcv_buf
     sum = sum + rcv_buf

  END DO

  WRITE(*,*) "PE", my_rank, ": Sum =", sum

  CALL MPI_Win_free(win)

  CALL MPI_Finalize()

END PROGRAM

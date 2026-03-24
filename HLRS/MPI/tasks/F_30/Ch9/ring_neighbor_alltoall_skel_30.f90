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
! Purpose: Using MPI_Neighbor_alltoall for ring communication. !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08
  IMPLICIT NONE

  INTEGER :: my_rank, size
  INTEGER :: right, left
  INTEGER :: i, sum
  INTEGER :: snd_buf_arr(0:1)
!            snd_buf = snd_buf_arr(0) ! ToDo ...
  INTEGER :: rcv_buf_arr(0:1)
!            rcv_buf = rcv_buf_arr(0)

  TYPE(MPI_Status)  :: status
  TYPE(MPI_Request) :: request
  TYPE(MPI_Comm) :: new_comm          
  INTEGER :: dims(1)
  LOGICAL :: reorder, periods(1)

  CALL MPI_Init()
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)

  dims(1) = size ; periods(1) = .TRUE. ; reorder = .TRUE.
  CALL MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, new_comm)
  CALL MPI_Comm_rank(new_comm, my_rank)
  CALL MPI_CART_SHIFT(new_comm, 0, 1, left, right)

  sum = 0
  snd_buf_arr(0) = my_rank ! ToDo ...
  ! snd_buf_arr(?) = -1000-my_rank ! only for test purpose, should be never used ! ToDo ...

  DO i = 1, size
     CALL MPI_Issend(snd_buf_arr(0), 1, MPI_INTEGER, right, 17, new_comm, request) ! ToDo ...
     CALL MPI_Recv(rcv_buf_arr(0), 1, MPI_INTEGER, left, 17, new_comm, status) ! ToDo ...
     CALL MPI_Wait(request, status) ! ToDo ...
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_SYNC_REG(snd_buf_arr) ! ToDo ...
!      CALL MPI_Neighbor_alltoall(__________________________________________________________________) ! ToDo ...
     snd_buf_arr(0) = rcv_buf_arr(0) ! ToDo ...
     sum = sum + rcv_buf_arr(0)
  END DO

  WRITE(*,*) "PE", my_rank, ": Sum =", sum

  CALL MPI_Finalize()

END PROGRAM

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
! Purpose: Creating a 1-dimens. topology with MPI_Cart_create  !
!          & using MPI_Cart_shift to calculate neighbor ranks  !
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

  TYPE(MPI_Comm) :: new_comm          
  INTEGER :: dims(1)
  LOGICAL :: reorder, periods(1)
! INTEGER :: coords(1) 

  CALL MPI_Init()
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)

! Prepare input arguments for creating a Cartesian topology.
  dims(1) = size
  periods(1) = .TRUE.
  reorder = .TRUE.

  CALL MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, new_comm)
! Get reordered my_rank (and coords if ndims>1)
  CALL MPI_Comm_rank(new_comm, my_rank)
! CALL MPI_Cart_coords(new_comm,my_rank, 1, coords) 

! Get nearest neighbour ranks.
  CALL MPI_Cart_shift(new_comm, 0, 1, left, right)

! The halo ring communication code from course chapter 4
  sum = 0
  snd_buf = my_rank
  DO i = 1, size
     CALL MPI_Issend(snd_buf, 1, MPI_INTEGER, right, 17, new_comm, request)
     CALL MPI_Recv  (rcv_buf, 1, MPI_INTEGER, left,  17, new_comm, status)
     CALL MPI_Wait(request, status)
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)
     snd_buf = rcv_buf
     sum = sum + rcv_buf
  END DO
  WRITE(*,*) "PE", my_rank, ": Sum =", sum
! WRITE(*,*) "PE", my_rank, " Coord =", coords(1), ": Sum =", sum

  CALL MPI_Finalize()

END PROGRAM

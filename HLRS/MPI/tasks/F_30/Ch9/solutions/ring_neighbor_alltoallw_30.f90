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
! Purpose: Using MPI_Neighbor_alltoallw for ring communication.!
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

  INTEGER(KIND=MPI_ADDRESS_KIND) :: snd_displs(2), rcv_displs(2)
  INTEGER :: snd_counts(2), rcv_counts(2)
  TYPE(MPI_Datatype) :: snd_types(2), rcv_types(2)

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)

  dims(1) = size
  periods(1) = .TRUE.
  reorder = .TRUE.
  CALL MPI_Cart_create(MPI_COMM_WORLD, 1, dims, periods, reorder, new_comm)
  CALL MPI_Comm_rank(new_comm, my_rank)
  CALL MPI_Cart_shift(new_comm, 0, 1, left, right)

  sum = 0
  snd_buf = my_rank
  rcv_buf = -1000 ! only for test purpose; should be overwritten by first MPI_Recv

  rcv_counts(1) = 1;  CALL MPI_Get_address(rcv_buf, rcv_displs(1));  rcv_types(1) = MPI_INTEGER
  rcv_counts(2) = 0;  rcv_displs(2) = 0;                             rcv_types(2) = MPI_INTEGER
  snd_counts(1) = 0;  snd_displs(1) = 0;                             snd_types(1) = MPI_INTEGER
  snd_counts(2) = 1;  CALL MPI_Get_address(snd_buf, snd_displs(2));  snd_types(2) = MPI_INTEGER

  DO i = 1, size
!    ... Substituted by MPI_Neighbor_alltoallw() :
!     CALL MPI_Issend(snd_buf, 1, MPI_INTEGER, right, 17, new_comm, request)
!     CALL MPI_Recv(rcv_buf, 1, MPI_INTEGER, left, 17, new_comm, status)
!     CALL MPI_Wait(request, status)
!     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)

     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)

     CALL MPI_Neighbor_alltoallw(MPI_BOTTOM, snd_counts, snd_displs, snd_types, &
                                 MPI_BOTTOM, rcv_counts, rcv_displs, rcv_types, new_comm)

     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(snd_buf)
     IF (.NOT.MPI_ASYNC_PROTECTS_NONBLOCKING) CALL MPI_F_sync_reg(rcv_buf)

     snd_buf = rcv_buf
     sum = sum + rcv_buf
  END DO

  WRITE(*,*) "PE", my_rank, ": Sum =", sum

  CALL MPI_Finalize()

END PROGRAM

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
! Purpose: A WRONG ring communication program                  *
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08

  IMPLICIT NONE

  INTEGER :: my_rank, size
  INTEGER :: right, left
  INTEGER :: i, sum
  INTEGER :: snd_buf, rcv_buf
  TYPE(MPI_Status)  :: status

  CALL MPI_Init()

  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)

  right = mod(my_rank+1,      size)
  left  = mod(my_rank-1+size, size)
!     ... this SPMD-style neighbor computation with modulo has the same meaning as:
!     right = my_rank + 1
!     IF (right .EQ. size) right = 0
!     left = my_rank - 1
!     IF (left .EQ. -1) left = size-1

  sum = 0
  snd_buf = my_rank
  DO i = 1, size
     IF (my_rank == 0) THEN
       CALL MPI_Recv(rcv_buf, 1, MPI_INTEGER, left,  17, MPI_COMM_WORLD, status)
       CALL MPI_Ssend(snd_buf, 1, MPI_INTEGER, right, 17, MPI_COMM_WORLD)
     ELSE
       CALL MPI_Ssend(snd_buf, 1, MPI_INTEGER, right, 17, MPI_COMM_WORLD)
       CALL MPI_Recv(rcv_buf, 1, MPI_INTEGER, left,  17, MPI_COMM_WORLD, status)
     ENDIF
     !  WRONG program, because the deadlock is resolved by a serialization
     !  which is for performance reasons wrong!
     !  And this program will still deadlock when running with only one process.
     snd_buf = rcv_buf
     sum = sum + rcv_buf
  END DO
  WRITE(*,*) "PE", my_rank, ": Sum =", sum

  CALL MPI_Finalize()

END PROGRAM

PROGRAM pingpong_bench

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
! Purpose: A program to try MPI_Ssend and MPI_Recv.            !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08

  IMPLICIT NONE

  INTEGER :: number_of_messages
  PARAMETER (number_of_messages=50)

  DOUBLE PRECISION :: start, finish, msg_transfer_time
  TYPE(MPI_Status) :: status
  REAL :: buffer(1)
  INTEGER :: i, my_rank

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)

  start = MPI_Wtime()
  DO i = 1, number_of_messages

     IF (my_rank .EQ. 0) THEN
        CALL MPI_Send(buffer, 1, MPI_REAL, 1, 17, MPI_COMM_WORLD)
        CALL MPI_Recv(buffer, 1, MPI_REAL, 1, 23, MPI_COMM_WORLD, status)
     ELSE IF (my_rank .EQ. 1) THEN
        CALL MPI_Recv(buffer, 1, MPI_REAL, 0, 17, MPI_COMM_WORLD, status)
        CALL MPI_Send(buffer, 1, MPI_REAL, 0, 23, MPI_COMM_WORLD)
     END IF

  END DO
  finish = MPI_Wtime()

  IF (my_rank .EQ. 0) THEN
     msg_transfer_time = ((finish - start) / (2 * number_of_messages)) * 1e6  ! in microsec
     WRITE(*,*) 'Time for one message:', msg_transfer_time, ' micro seconds'
  END IF

  CALL MPI_Finalize()

END PROGRAM

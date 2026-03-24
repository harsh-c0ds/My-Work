PROGRAM pingpong

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
! Authors: Rolf Rabenseifner (HLRS)                            !
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

  TYPE(MPI_Status) :: status
  REAL :: temp, mass
  INTEGER :: my_rank

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)

     IF (my_rank .EQ. 0) THEN
        temp = 100 ;  mass = 2.5
        CALL MPI_Send(temp, 1, MPI_REAL, 1, 17, MPI_COMM_WORLD)
        CALL MPI_Send(mass, 1, MPI_REAL, 1, 18, MPI_COMM_WORLD)
     ELSE IF (my_rank .EQ. 1) THEN
        CALL MPI_Recv(mass, 1, MPI_REAL, 0, 18, MPI_COMM_WORLD, status)
        WRITE(*,*) 'I am process', my_rank, ' and mass=', mass
        CALL MPI_Recv(temp, 1, MPI_REAL, 0, 17, MPI_COMM_WORLD, status)
        WRITE(*,*) 'I am process', my_rank, ' and temp=', temp
     END IF

  CALL MPI_Finalize()

END PROGRAM

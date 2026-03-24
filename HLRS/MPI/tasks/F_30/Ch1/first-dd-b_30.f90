PROGRAM first_example

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
! Purpose: A first MPI example calculating the subdomain size  !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08

  IMPLICIT NONE

  INTEGER :: n               ! application-related data
  DOUBLE PRECISION :: result ! application-related data
  INTEGER :: my_rank, num_procs, rank  ! MPI-related data
  INTEGER :: sub_n, sub_start ! size and starting index of "my" sub domain
  INTEGER :: num_larger_procs

  CALL MPI_Init()

  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, num_procs)

  IF (my_rank == 0) THEN  
    !  reading the application data "n" from stdin only by process 0:
    WRITE(*,*) "Enter the number of elements (n):"
    READ(*,*) n
  ENDIF
  !  broadcasting the content of variable "n" in process 0 
  !  into variables "n" in all other processes:
  CALL MPI_Bcast(n, 1, MPI_INTEGER, 0, MPI_COMM_WORLD)

  !  Calculating the number of elements of my subdomain: sub_n
  !  Calculating the start index sub_start within 0..n-1 
  !  or sub_start = -1 and sub_n = 0 if there is no element

  !  The following algorithm divided 5 into 2 + 1 + 1 + 1
  sub_n = n / num_procs  ! = rounding_off(n/num_procs)
  num_larger_procs = n - num_procs*sub_n  ! = #procs with sub_n+1 elements
  IF (my_rank < num_larger_procs) THEN
    sub_n = sub_n + 1
    sub_start = 0 + my_rank * sub_n
  ELSE IF (sub_n > 0) THEN
    sub_start = 0 + num_larger_procs + my_rank * sub_n
  ELSE
    !  this process has only zero elements
    sub_start = -1
    sub_n = 0
  ENDIF

  WRITE(*,'(A,I3,A,I3,A,I5,A,I5,A,I5)') &
   &        'I am process ', my_rank, ' out of ', num_procs, &
   &        ', responsible for the ', sub_n, ' elements with indexes ', sub_start, ' .. ', sub_start+sub_n-1

  CALL MPI_Finalize()

END PROGRAM

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
! Purpose: Gathering data from all processes, MPI_IN_PLACE     !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08

  IMPLICIT NONE

  INTEGER :: n               ! application-related data
  DOUBLE PRECISION :: result ! application-related data
  DOUBLE PRECISION, ALLOCATABLE, DIMENSION (:) :: result_array
  INTEGER :: my_rank, num_procs, rank, root  ! MPI-related data

  CALL MPI_Init()

  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, num_procs)
  root = num_procs - 1

  !  doing some application work in each process, e.g.:
  result = 100.0 + 1.0 * my_rank
  WRITE(*,'(A,I3,A,I3,A,I2,A,I5,A,F9.2)') &
   &        'I am process ', my_rank, ' out of ', num_procs, &
   &        ' handling the ', my_rank, 'th part of n=', n, ' elements, result=', result

  IF (my_rank == root) THEN  
    ALLOCATE(result_array(0:num_procs-1))
  ENDIF

  IF (my_rank == root) THEN
    result_array(root) = result
    CALL MPI_Gather(MPI_IN_PLACE,0,MPI_DATATYPE_NULL, result_array,1,MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD)
  ELSE
    CALL MPI_Gather(result,1,MPI_DOUBLE_PRECISION, result_array,1,MPI_DOUBLE_PRECISION, root, MPI_COMM_WORLD)
  ENDIF
  IF (my_rank == root) THEN  
    DO rank=0, num_procs-1
      WRITE(*,'(A,I3,A,I3,A,F9.2)') &
       &      'I''m proc ', root, ': result of process ', rank, ' is ', result_array(rank) 
    END DO
  ENDIF

  CALL MPI_Finalize()

END PROGRAM

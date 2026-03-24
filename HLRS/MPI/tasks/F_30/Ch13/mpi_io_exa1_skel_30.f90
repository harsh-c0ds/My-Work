PROGRAM mpi_io_test

!***********************************************************************
!
! This file has been written as a sample solution to an exercise in a 
! course given at the HLRS www.hlrs.de . It is made
! freely available with the understanding that every copy of this file
! must include this header and that HLRS takes no responsibility for
! the use of the enclosed teaching material.
!
! Authors:    Rolf Rabenseifner
!
! Contact:    rabenseifner@hlrs.de 
!
! Purpose:    A program to test parallel file I/O with MPI.
!
! Contents:   F source code.
!
!***********************************************************************

  USE mpi_f08
  IMPLICIT NONE

  INTEGER :: ierror, my_rank, size, i

  TYPE(MPI_File) :: fh
  INTEGER (_____) :: offset
  TYPE(MPI_Status) :: status

  CHARACTER :: buf 

  CALL MPI_Init(ierror)

  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierror)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size, ierror)

  CALL MPi_File_open(MPI_COMM_WORLD, 'my_test_file',      &
     &               IOR(MPI_MODE_____, MPI_MODE_____), &
     &               MPI_INFO_NULL, fh, ierror) 
 
  DO i=1,10
    buf = CHAR( ICHAR('0') + my_rank ) 
    offset = _____
    CALL MPI_File_write_at(fh, offset, buf, _____, _____,  &
     &                     status, ierror) 
  END DO 

  CALL MPI_File_close(fh, ierror)

  WRITE(*,*) 'PE=',my_rank

  CALL MPI_Finalize(ierror)

END

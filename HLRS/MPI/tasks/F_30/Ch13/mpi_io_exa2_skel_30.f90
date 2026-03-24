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

  INTEGER :: ndims, array_of_sizes(1), array_of_subsizes(1)
  INTEGER :: array_of_starts(1), order
  TYPE(MPI_File) :: fh
  TYPE(MPI_Datatype) :: etype
  TYPE(MPI_Datatype) :: filetype
  INTEGER (____) :: disp 
  TYPE(MPI_Status) :: status

  CHARACTER :: buf 

  CALL MPI_INIT(ierror)

  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank, ierror)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size, ierror)

  etype = MPI_CHARACTER
  ndims = ____
  array_of_sizes(1)    = ____
  array_of_subsizes(1) = ____
  array_of_starts(1)   = ____
  order = MPI_ORDER_FORTRAN 
  CALL MPI_Type_create_subarray(ndims, array_of_sizes,   &
     &                         array_of_subsizes, array_of_starts,   &
     &                         order, etype, filetype, ierror)
  CALL MPI_Type_____

  CALL MPI_File_open(MPI_COMM_WORLD, 'my_test_file',    &
     &               IOR(MPI_MODE_____, MPI_MODE_____),   &
     &               MPI_INFO_NULL, fh, ierror) 
 
  disp = ____
  CALL MPI_File_set_view(fh, disp, etype, filetype, 'native',   &
     &                   MPI_INFO_NULL, ierror) 

  DO I=1,3
    buf = CHAR( ICHAR('a') + my_rank ) 
    CALL MPI_File_write(fh, buf, ____, ____, status, ierror) 
  END DO 

  CALL MPI_File_close(fh, ierror)

  WRITE(*,*) 'PE=',my_rank

  CALL MPI_Finalize(ierror)

  STOP
END

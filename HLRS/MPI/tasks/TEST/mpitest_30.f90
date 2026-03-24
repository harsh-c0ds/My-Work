PROGRAM mpitest

!==============================================================!
!                                                              !
! This file has been written as a sample solution to an        !
! exercise in a course given at the High Performance           !
! Computing Centre Stuttgart (HLRS).                           !
! It is made freely available with the understanding that      !
! every copy of this file must include this header and that    !
! HLRS take no responsibility for the use of the               !
! enclosed teaching material.                                  !
!                                                              !
! Authors: Rolf Rabenseifner (HLRS)                            !
!                                                              !
! Contact: rabenseifner@hlrs.de                                !
!                                                              ! 
! Purpose: Check version of the MPI library and include file   !
!                                                              !
! Contents: Fortran-Source                                     !
!                                                              !
!==============================================================!


  USE mpi_f08
  IMPLICIT NONE

  INTEGER :: my_rank, size
  INTEGER :: version, subversion

  CALL MPI_Init()
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size)
  CALL MPI_Get_version(version, subversion)
 
  IF (my_rank == 0) THEN
    IF (size > 1) THEN
      WRITE(*,'(A,I3,A)') &
       & 'Successful first MPI test executed in parallel on ', size, ' processes.'
    ELSE
      WRITE(*,'(A)') 'Caution: This MPI test is executed only on one MPI process, i.e., sequentially!'
    ENDIF
    WRITE(*,'(A,I2,A,I1,A)') & 
     & 'Your installation supports MPI standard version ', version, '.', subversion, '.'
  ENDIF

  CALL MPI_Finalize()

END

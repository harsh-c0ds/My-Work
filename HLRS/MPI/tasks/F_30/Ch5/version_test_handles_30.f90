PROGRAM version

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
! Purpose: Check for TYPE(MPI_Comm)                            !
!                                                              !
! Contents: Fortran-Source                                     !
!                                                              !
!==============================================================!


  USE mpi_f08
  IMPLICIT NONE

  INTEGER lib_version, lib_subversion, lib_version_max, ierror

  TYPE(MPI_Comm)   :: f08_comm1, f08_comm2
  INTEGER          :: f_comm1, f_comm2

  CALL MPI_Init()

  CALL MPI_Get_version(VERSION=lib_version, SUBVERSION=lib_subversion)
  WRITE(*,*) 'Version: Library:        ', lib_version, lib_subversion
  WRITE(*,*) 'Version: mpi_f08 module: ', MPI_VERSION, MPI_SUBVERSION
  WRITE(*,*) 'MPI_ASYNC_PROTECTS_NONBLOCKING: ', MPI_ASYNC_PROTECTS_NONBLOCKING
  WRITE(*,*) 'MPI_SUBARRAYS_SUPPORTED       : ', MPI_SUBARRAYS_SUPPORTED

  f08_comm1 = MPI_COMM_WORLD
  f08_comm2 = MPI_COMM_WORLD
  IF ((f08_comm1 .NE. f08_comm2) .OR. (f08_comm1 /= f08_comm2)) THEN
   WRITE(*,*) 'The overloaded .NE and /= operators for MPI_Comm do not work'
  ENDIF

  CALL MPI_Finalize()

END

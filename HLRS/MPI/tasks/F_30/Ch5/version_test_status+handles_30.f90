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
! Purpose: Check for TYPE(MPI_Status) and TYPE(MPI_Comm)       !
!                                                              !
! Contents: Fortran-Source                                     !
!                                                              !
!==============================================================!


  USE mpi_f08
  IMPLICIT NONE

  INTEGER lib_version, lib_subversion, lib_version_max, ierror

  TYPE(MPI_Status) :: f08_status
  INTEGER          :: f_status(MPI_STATUS_SIZE)
  TYPE(MPI_Comm)   :: f08_comm1, f08_comm2
  INTEGER          :: f_comm1, f_comm2
  INTEGER          :: f_count, f08_count

  CALL MPI_Init()

  CALL MPI_Get_version(VERSION=lib_version, SUBVERSION=lib_subversion)
  WRITE(*,*) 'Version: Library:        ', lib_version, lib_subversion
  WRITE(*,*) 'Version: mpi_f08 module: ', MPI_VERSION, MPI_SUBVERSION
  WRITE(*,*) 'MPI_ASYNC_PROTECTS_NONBLOCKING: ', MPI_ASYNC_PROTECTS_NONBLOCKING
  WRITE(*,*) 'MPI_SUBARRAYS_SUPPORTED       : ', MPI_SUBARRAYS_SUPPORTED

  f08_status%MPI_SOURCE = 1
  f08_status%MPI_TAG = 17
  f08_status%MPI_ERROR = MPI_ERR_IN_STATUS
  f08_count = 300 
  CALL MPI_Status_set_elements(f08_status, MPI_INTEGER, f08_count)
  CALL MPI_Status_f082f(f08_status, f_status)
  CALL MPI_Status_f2f08(f_status, f08_status)
  IF (f_status(MPI_SOURCE) /= f08_status%MPI_SOURCE) THEN
    WRITE(*,*) 'Error in f_status: f_status(MPI_SOURCE)=',f_status(MPI_SOURCE), &
     &                      ' .NE. f08_status%MPI_SOURCE=',f08_status%MPI_SOURCE
  ENDIF
  IF (f_status(MPI_TAG) /= f08_status%MPI_TAG) THEN
    WRITE(*,*) 'Error in f_status: f_status(MPI_TAG)=',f_status(MPI_TAG), &
     &                      ' .NE. f08_status%MPI_TAG=',f08_status%MPI_TAG
  ENDIF
  IF (f_status(MPI_ERROR) /= f08_status%MPI_ERROR) THEN
    WRITE(*,*) 'Error in f_status: f_status(MPI_ERROR)=',f_status(MPI_ERROR), &
     &                      ' .NE. f08_status%MPI_ERROR=',f08_status%MPI_ERROR
  ENDIF

  f08_comm1 = MPI_COMM_WORLD
  f08_comm2 = MPI_COMM_WORLD
  IF ((f08_comm1 .NE. f08_comm2) .OR. (f08_comm1 /= f08_comm2)) THEN
   WRITE(*,*) 'The overloaded .NE and /= operators for MPI_Comm do not work'
  ENDIF

  CALL MPI_Finalize()

END

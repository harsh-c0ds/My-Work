PROGRAM data_rep

! ************************************************************************** !
!                                                                            !
! data-replication in distributed and shared memory                          !
! program (C source code).                                                   !
!                                                                            !
! - the skeleton bcasts the data to all processes                            !
! - solution: rank_world == 0 puts the data                                  !
!             into the shared memory of node 0 ,                             !
!             rank_world == 0 bcasts the data to one of the processes        !
!             of each of the other nodes, only ,                             !
!             i.e., to all the other rank_shm==0 processes                   !
!                                                                            !
! - Course material: Introduction to Hybrid Programming in HPC               !
!                                                                            !
!                    It is made freely available with the understanding that !
!                    every copy must include this header and that            !
!                    the authors as well as VSC and TU Wien                  !
!                    take no responsibility for the use of this program.     !
!                                                                            !
!        (c) 01/2019 Irene Reichl (VSC Team, TU Wien)                        !
!                    irene.reichl@tuwien.ac.at                               !
!                                                                            !
!      vsc3:  module load intel/18 intel-mpi/2018                            !
!      vsc3:  mpiicc -o data-rep_solution data-rep_solution.c                !
!                                                                            !
! ************************************************************************** !

  USE mpi_f08
  USE, INTRINSIC :: ISO_C_BINDING

  IMPLICIT NONE

  TYPE(MPI_Datatype), PARAMETER :: arrDataType = MPI_INTEGER8

  INTEGER, PARAMETER :: arrSize = 16*1600000
  INTEGER*8, DIMENSION(:), ALLOCATABLE :: arr

  INTEGER :: it
  INTEGER :: rank_world, size_world
  INTEGER :: i
  INTEGER*8 :: sum

!===> 1 <===
  CALL MPI_Init()

  CALL MPI_Comm_rank(MPI_COMM_WORLD, rank_world)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size_world)

!===> 2 <===
  ALLOCATE(arr(1:arrSize))

!===> 3 <===
 DO it=1,3

!/* only rank_world=0 initializes the array arr                 */
   IF( rank_world == 0 ) THEN
     DO i=1, arrSize
       arr(i) = i + it
     ENDDO
   ENDIF

!===> 4 <===
   CALL MPI_Bcast(arr, arrSize, arrDataType, 0, MPI_COMM_WORLD)

!/* Now, all arrays are filled with the same content. */

!===> 5 <===
   sum = 0
   DO i=1, arrSize
     sum = sum + arr(i)
   ENDDO
  
!===> 6 <===
   ! TEST: To minimize the output, we print only from 3 process
   IF ( (rank_world == 0) .OR. (rank_world == 1) .OR. (rank_world == size_world - 1) ) THEN
     WRITE(*,*)' it=',it, ' rank( world=',rank_world,')', &
     &                    ' sum(i=',1+it,'...i=',arrSize+it,') = ',sum 
   ENDIF
 ENDDO

!===> 7 <===
  DEALLOCATE(arr)

  CALL MPI_Finalize()
END

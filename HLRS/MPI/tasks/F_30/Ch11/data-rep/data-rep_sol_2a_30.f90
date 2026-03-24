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
!  INTEGER*8, DIMENSION(:), POINTER :: arr

  INTEGER :: it
  INTEGER :: rank_world, size_world
  INTEGER :: i
  INTEGER*8 :: sum

!===> 1 <===
!/* TO DO --> comm_shm, size_shm, rank_shm, shared window win, *shm_buf_ptr
! * individualShmSize 
! 
! * MPI_Win_shared_query --> *arr 
! 
! * comm_head; size_head, rank_head;
! */
  TYPE(MPI_Comm) :: comm_shm 
  INTEGER :: size_shm, rank_shm
  !____________ :: win
  INTEGER :: individualShmSize
  TYPE(C_PTR) :: arr_ptr, shm_buf_ptr
  INTEGER(KIND=MPI_ADDRESS_KIND) :: arrDataTypeSize, lb, ShmByteSize

! /* output MPI_Win_shared_query */
  INTEGER(kind=MPI_ADDRESS_KIND) :: arrSize_
  INTEGER :: disp_unit

  INTEGER :: color
  !_____________ :: comm_head
  INTEGER :: size_head, rank_head

  CALL MPI_Init()

  CALL MPI_Comm_rank(MPI_COMM_WORLD, rank_world)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size_world)

!===> 2 <===
!/*  TO DO: 
! *  substitute the following malloc
! */
  ALLOCATE(arr(1:arrSize))
!/*  by
! *   MPI_Comm_split_type        ---> create shm islands
! *   MPI_Win_allocate_shared    ---> create shm windows
! *                                   such that rank_shm = 0 has shared memory portion equal to size of arr
! *                                   and rank_shm != 0 have shared memory portion equal to zero
! *   MPI_Win_shared_query       ---> get starting address of rank_shm = 0 's shared memory portion
! */

! /* Create --> shared memory islands and --> shared memory window inside */
! /*           -->    comm_shm         and      -->    win                 */

!                                                            key=0
  CALL MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm_shm)
  CALL MPI_Comm_size(comm_shm, size_shm)
  CALL MPI_Comm_rank(comm_shm, rank_shm) 

  ! TEST: To minimize the output, we print only from 3 process per SMP node 
  IF ( (rank_shm == 0) .OR. (rank_shm == 1) .OR. (rank_shm == size_shm - 1) ) THEN
    WRITE(*,*) ' rank( world=',rank_world, ' shm=',rank_shm,')'
  ENDIF
  IF (rank_world == 0) WRITE(*,*) 'ALL finalize and return !!!'; CALL MPI_Finalize(); STOP

  ! instead of:  ALLOCATE(arr(1:arrSize))

  ! IF ( _____________ ) THEN
  !   individualShmSize = arrSize
  ! ELSE
  !   individualShmSize = 0
  ! ENDIF
  ! CALL MPI_Type_get_extent(arrDataType, lb, arrDataTypeSize)
  ! ShmByteSize = individualShmSize * arrDataTypeSize
  ! disp_unit = arrDataTypeSize
  ! CALL MPI_Win_allocate_shared( ShmByteSize, disp_unit, _____________, ________, shm_buf_ptr, win )
  !  /* shm_buf_ptr is not used because it is only available in process rank_shm==0 */
  ! CALL MPI_Win_shared_query( ___,__, arrSize_, disp_unit, arr_ptr )
  ! CALL C_F_POINTER(arr_ptr, arr, (/arrSize/) )
  ! IF ( (rank_shm == 0) .OR. (rank_shm == 1) .OR. (rank_shm == size_shm - 1) ) THEN
  !   WRITE(*,*) ' process=',rank_world, ' arrSize=',arrSize, ' arrSize_=',arrSize_ 
  ! ENDIF

!/* TO DO: Create communicator comm_head with MPI_Comm_split  -->  including all the rank_shm == 0 processes.
! * Only the rank_shm==0 processes should have a color, all others have color=MPI_UNDEFINED 
! * this meands that only the head processes are in the communicator, the other processes are not included,
! * i.e.,  their comm_head is MPI_COMM_NULL.
! * The advantage of excluding all the other ranks compared to assigning color = 1 is that the communicator 
! * is a smaller entity, hence, it allows for better scaling.
! *
! * Note that subsequent MPI-calls within comm_head must only be performed for processes on comm_head
! * -->   if( comm_head != MPI_COMM_NULL ) 
! *
! * Choose key value in MPI_Comm_split_type(.... &comm_shm) and MPI_Comm_split(.... &comm_head) 
! * such that the rank_world==0 process has also rank==0 within its comm_shm and within comm_head.
! */
  
 ! color=MPI_UNDEFINED
 ! IF (rank_shm==0) THEN
 !   color = 0
 ! ENDIF

 ! CALL MPI_Comm_split(...)
 !
 !  ... size_head)
 !  ... rank_head)

!===> 3 <===
 DO it=1,3

!/* TO DO: the following line writes on the shared memory.
! *        take care for correct synchronization with memory fences !   
! *        The MPI_Win_fence starts the write epoch for all rank_shm==0 processes
! */
!/* only rank_world=0 initializes the array arr                 */
   IF( rank_world == 0 ) THEN 
     DO i=1, arrSize
       arr(i) = i + it
     ENDDO
   ENDIF

!===> 4 <===
!/* Instead of all processes in MPI_COMM_WORLD, now only the heads of the 
! * shared memory islands communicate (using comm_head).
! * Since we used key=0 in both MPI_Comm_split(...), process rank_world = 0
! * - is also rank 0 in comm_head
! * - and rank 0 in comm_shm in the color it belongs to.                              */

   CALL MPI_Bcast(arr, arrSize, arrDataType, 0, MPI_COMM_WORLD)

!/* Now, all arrays are filled with the same content. */

!===> 5 <===
!/* TO DO: the following lines read from the shared memory and compute the sum.
! *        take care for correct synchronization with memory fences !   
! *        The MPI_Win_fence now has to switch from the write epoch to the read epoch
! *        allowing reading arr by all processes belonging to their comm_shm.
! */

   sum = 0
   DO i=1, arrSize
     sum = sum + arr(i)
   ENDDO
  
!===> 6 <===
   ! TEST: To minimize the output, we print only from 3 process per SMP node 
   IF ( (rank_world == 0) .OR. (rank_world == 1) .OR. (rank_world == size_world - 1) ) THEN
     WRITE(*,*)' it=',it, ' rank( world=',rank_world,')', &
     &                    ' sum(i=',1+it,'...i=',arrSize+it,') = ',sum 
   ENDIF
   ! /* already prepared for the shared memory solution: */
   ! IF ( (rank_shm == 0) .OR. (rank_shm == 1) .OR. (rank_shm == size_shm - 1) ) THEN
   !   WRITE(*,*)' it=',it, ' rank( world=',rank_world, ' shm=',rank_shm, ' head=',rank_head,')', &
   !   &                    ' sum(i=',1+it,'...i=',arrSize+it,') = ',sum 

 ENDDO

!===> 7 <===
!/* TO DO: there is no malloc and therefore no free of arr.
! *        instead free the shared memory and
! *        guarantee that all operations on the shared memory have been completed before the shared memory is freed
! */

  DEALLOCATE(arr)

  CALL MPI_Finalize()
END

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
!  INTEGER*8, DIMENSION(:), ALLOCATABLE :: arr
  INTEGER*8, DIMENSION(:), POINTER :: arr

  INTEGER :: it
  INTEGER :: rank_world, size_world
  INTEGER :: i
  INTEGER*8 :: sum

!===> 1 <===
  TYPE(MPI_Comm) :: comm_shm 
  INTEGER :: size_shm, rank_shm
  TYPE(MPI_Win) :: win
  INTEGER :: individualShmSize
  TYPE(C_PTR) :: arr_ptr, shm_buf_ptr
  INTEGER(KIND=MPI_ADDRESS_KIND) :: arrDataTypeSize, lb, ShmByteSize

! /* output MPI_Win_shared_query */
  INTEGER(kind=MPI_ADDRESS_KIND) :: arrSize_
  INTEGER :: disp_unit

  INTEGER :: color
  TYPE(MPI_Comm) :: comm_head
  INTEGER :: size_head, rank_head

  CALL MPI_Init()

  CALL MPI_Comm_rank(MPI_COMM_WORLD, rank_world)
  CALL MPI_Comm_size(MPI_COMM_WORLD, size_world)

!===> 2 <===
! /* Create --> shared memory islands and --> shared memory window inside */
! /*           -->    comm_shm         and      -->    win                 */

!                                                            key=0
  CALL MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, comm_shm)
  CALL MPI_Comm_size(comm_shm, size_shm)
  CALL MPI_Comm_rank(comm_shm, rank_shm) 

  ! instead of:  ALLOCATE(arr(1:arrSize))
  IF ( rank_shm == 0 ) THEN
    individualShmSize = arrSize
  ELSE
    individualShmSize = 0
  ENDIF
  CALL MPI_Type_get_extent(arrDataType, lb, arrDataTypeSize)
  ShmByteSize = individualShmSize * arrDataTypeSize
  disp_unit = arrDataTypeSize
  CALL MPI_Win_allocate_shared( ShmByteSize, disp_unit, MPI_INFO_NULL, comm_shm, shm_buf_ptr, win )
  !  /* shm_buf_ptr is not used because it is only available in process rank_shm==0 */
  CALL MPI_Win_shared_query( win, 0, arrSize_, disp_unit, arr_ptr )
  CALL C_F_POINTER(arr_ptr, arr, (/arrSize/) )

! /* Create communicator including all the rank_shm = 0               */
! /* with the MPI_Comm_split: in color 0 all the rank_shm = 0 , 
!  * all other ranks are color = 1                                        */

  color=MPI_UNDEFINED
  IF (rank_shm==0) THEN
    color = 0
  ENDIF

!                                      key = 0
  CALL MPI_Comm_split(MPI_COMM_WORLD, color, 0, comm_head)
  rank_head = -1 ! // only used in the print statements to differentiate unused rank==-1 from used rank==0
  IF (comm_head /= MPI_COMM_NULL) THEN ! if( color == 0 ) // rank is element of comm_head, i.e., it is head of one of the islands in comm_shm
    CALL MPI_Comm_size(comm_head, size_head)
    CALL MPI_Comm_rank(comm_head, rank_head)
  ENDIF

!===> 3 <===
 DO it=1,3

!/* only rank_world=0 initializes the array arr                 */
!/* all rank_shm=0 start the write epoch: writing arr to their shm */
   CALL MPI_Win_fence(0, win) ! workaround: no assertions
   IF( rank_world == 0 ) THEN ! /* from those rank_shm=0 processes, only rank_world==0 fills arr */
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

   IF( comm_head /= MPI_COMM_NULL ) THEN ! // if( color == 0 )
     CALL MPI_Bcast(arr, arrSize, arrDataType, 0, comm_head)
!    /* with this Bcast, all other rank_shm=0 processes write the data into their arr */
   ENDIF

!/* Now, all arrays are filled with the same content. */

!===> 5 <===
   CALL MPI_Win_fence(0, win) ! // after the fence all processes start a read epoch // workaround: no assertions: 0

!/* Now, all other ranks in the comm_sm shared memory islands are allowed to access their shared memory array. */
!/* And all ranks rank_sm access the shared mem in order to compute sum  */
   sum = 0
   DO i=1, arrSize
     sum = sum + arr(i)
   ENDDO
  
!===> 6 <===
  ! TEST: To minimize the output, we print only from 3 process per SMP node 
  IF ( (rank_shm == 0) .OR. (rank_shm == 1) .OR. (rank_shm == size_shm - 1) ) THEN
    WRITE(*,*)' it=',it, ' rank( world=',rank_world, ' shm=',rank_shm, ' head=',rank_head,')', &
    &                    ' sum(i=',1+it,'...i=',arrSize+it,') = ',sum 
  ENDIF

 ENDDO

!===> 7 <===
  CALL MPI_Win_fence(0, win) ! // free destroys the shm. fence to guarantee that read epoch has been finished // workaround: no assertions: 0
  CALL MPI_Win_free(win)
  ! DEALLOCATE(arr) ! arr is automatically deallocated by MPI_Win_free(win)

  CALL MPI_Finalize()
END

program exa_block 

  USE mpi_f08
  implicit none 

  integer i, ierror
  integer gsize, lsize, psize, rank, nblocks
  parameter (gsize=20000000)
 
! real*8 garray(gsize)
!HPF$   PROCESSORS   P(psize)
!HPF$   DISTRIBUTE garray(BLOCK)
! real*8 larray(lsize)

! inline function definition for 
! division with rounding to next upper integer 
  integer   ii,jj, UPPER_DIV 
  UPPER_DIV(ii,jj) = (ii+jj-1) / jj

  CALL MPI_INIT(ierror)
  CALL MPI_COMM_SIZE(MPI_COMM_WORLD, psize, ierror)
  CALL MPI_COMM_RANK(MPI_COMM_WORLD, rank,  ierror) 
 
  lsize = UPPER_DIV(gsize, psize) 
  nblocks = UPPER_DIV(gsize, lsize) 
  if(rank.GE.nblocks)   lsize = 0 
  if(rank.EQ.nblocks-1) lsize = gsize - (nblocks-1)*lsize
   
! write(*,*)'rank=',rank,'  g=',gsize,'  l=',lsize,'  p=',psize
  if (rank.eq.0) write(*,*) 'gsize=',gsize, ' on',psize,'PEs'

  CALL sub(rank,gsize,lsize,psize)

  CALL MPI_FINALIZE(ierror)

end 

!-----------------------------------------------------------------------

subroutine sub(rank,gsize,lsize,psize)

  USE mpi_f08
  implicit none 

  integer rank,gsize,lsize,psize
 
  TYPE(MPI_Datatype) :: darray_type
  TYPE(MPI_Status) :: status
  TYPE(MPI_File) :: fh
  INTEGER :: i, ierror
  INTEGER :: distribs, dargs
  INTEGER (KIND=MPI_OFFSET_KIND) :: disp, lb, lext 
  LOGICAL first
 
  REAL*8 :: larray(lsize)

  DOUBLE PRECISION :: start_time, end_time 

  distribs = MPI_DISTRIBUTE_BLOCK
  dargs    = MPI_DISTRIBUTE_DFLT_DARG

  CALL MPI_TYPE_CREATE_DARRAY(psize,rank,1,[gsize],[distribs],      &
     &                         [dargs], [psize], MPI_ORDER_FORTRAN, &
     &                         MPI_REAL8, darray_type,ierror)
  CALL MPI_TYPE_COMMIT(darray_type,ierror) 

  CALL MPI_TYPE_GET_EXTENT(MPI_REAL8, lb, lext, ierror) 
 
! --------------- 
 
  do i=1,lsize
   larray(i) = 100*i + rank
  enddo
  IF (rank == psize/2) THEN
    write (*,*) 'to test the read-verification, in process ',rank,' at offest ',lsize/4, &
       & ' -999 is stored instead of', larray(lsize/4)
    larray(lsize/4) = -999  ! to TEST whether the read-verification really works!!!!!!
  ENDIF 

  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  start_time = MPI_WTIME() 

  CALL MPI_FILE_OPEN(MPI_COMM_WORLD, 'exa_block_testfile',  &
     &               IOR(MPI_MODE_CREATE, MPI_MODE_WRONLY), &
     &               MPI_INFO_NULL, fh, ierror)
  disp = 0 
  CALL MPI_FILE_SET_VIEW(fh, disp, MPI_REAL8, darray_type, &
     &               'native'    ,MPI_INFO_NULL,ierror)
 
  CALL MPI_FILE_WRITE_ALL(fh, larray, lsize, MPI_REAL8, status,ierror)

  CALL MPI_FILE_CLOSE(fh,ierror)
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  end_time = MPI_WTIME() 

! write(*,*)'rank=',rank,'  done'
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  if (rank.eq.0) then
    write(*,*) 'WRITE_ALL in', end_time - start_time, 'sec', &
     &  ' ==> ', 1e-6*gsize*lext/(end_time-start_time), 'MB/s'  
  endif 

! --------------- 
 
  do i=1,lsize
   larray(i) = 0
  enddo
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  start_time = MPI_WTIME() 

  CALL MPI_FILE_OPEN(MPI_COMM_WORLD, 'exa_block_testfile',  &
     &               MPI_MODE_RDONLY, &
     &               MPI_INFO_NULL, fh, ierror)
  disp = 0 
  CALL MPI_FILE_SET_VIEW(fh, disp, MPI_REAL8, darray_type, &
     &               'native'    ,MPI_INFO_NULL,ierror)
 
  CALL MPI_FILE_READ_ALL(fh, larray, lsize, MPI_REAL8, status,ierror)

  CALL MPI_FILE_CLOSE(fh,ierror)
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  end_time = MPI_WTIME() 

! write(*,*)'rank=',rank,'  done'
  
  first = .TRUE.
  DO i=1,lsize
   IF ( larray(i) /=  100*i + rank ) THEN
     IF (first) THEN
       write(*,*) 'RANK=', rank, ' wrong data at offset=', i, ':', larray(i), '/=', 100*i + rank  
     ENDIF
     first = .FALSE.
   ENDIF
  ENDDO
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  if (rank.eq.0) then
    write(*,*) 'READ_ALL  in', end_time - start_time, 'sec',   &
     &  ' ==> ', 1e-6*gsize*lext/(end_time-start_time), 'MB/s'  
  endif 

! --------------- 
 
  do i=1,lsize
   larray(i) = 200*i + rank
  enddo
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  start_time = MPI_WTIME() 

  CALL MPI_FILE_OPEN(MPI_COMM_WORLD, 'exa_block_testfile',  &
     &               IOR(MPI_MODE_CREATE, MPI_MODE_WRONLY), &
     &               MPI_INFO_NULL, fh, ierror)
  disp = 0 
  CALL MPI_FILE_SET_VIEW(fh, disp, MPI_REAL8, darray_type, &
     &               'native'    ,MPI_INFO_NULL,ierror)
 
  CALL MPI_FILE_WRITE(fh, larray, lsize, MPI_REAL8, status,ierror)

  CALL MPI_FILE_CLOSE(fh,ierror)
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  end_time = MPI_WTIME() 

! write(*,*)'rank=',rank,'  done'
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  if (rank.eq.0) then
    write(*,*) 'WRITE     in', end_time - start_time, 'sec',   &
     &  ' ==> ', 1e-6*gsize*lext/(end_time-start_time), 'MB/s'  
  endif 

! --------------- 
 
  do i=1,lsize
   larray(i) = 0
  enddo
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  start_time = MPI_WTIME() 

  CALL MPI_FILE_OPEN(MPI_COMM_WORLD, 'exa_block_testfile',  &
     &               MPI_MODE_RDONLY, &
     &               MPI_INFO_NULL, fh, ierror)
  disp = 0 
  CALL MPI_FILE_SET_VIEW(fh, disp, MPI_REAL8, darray_type, &
     &               'native'    ,MPI_INFO_NULL,ierror)
 
  CALL MPI_FILE_READ(fh, larray, lsize, MPI_REAL8, status,ierror)

  CALL MPI_FILE_CLOSE(fh,ierror)
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  end_time = MPI_WTIME() 

! write(*,*)'rank=',rank,'  done'
  
  first = .TRUE.
  DO i=1,lsize
   IF ( larray(i) /=  200*i + rank ) THEN
     IF (first) THEN
       write(*,*) 'RANK=', rank, ' wrong data at offset=', i, ':', larray(i), '/=', 200*i + rank  
     ENDIF
     first = .FALSE.
   ENDIF
  ENDDO
 
  CALL MPI_BARRIER(MPI_COMM_WORLD, ierror) ! only for benchmarking
  if (rank.eq.0) then
    write(*,*) 'READ      in', end_time - start_time, 'sec',   &
     &  ' ==> ', 1e-6*gsize*lext/(end_time-start_time), 'MB/s'  
  endif 

! --------------- 
 
  return 
end 

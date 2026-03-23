program hello

integer :: my_rank, size
logical :: with_openmp = .FALSE.

!$ integer OMP_GET_THREAD_NUM, OMP_GET_NUM_THREADS

!$ with_openmp = .TRUE.
IF (with_openmp) THEN

!$OMP PARALLEL PRIVATE(my_rank, size)
!$ my_rank = OMP_GET_THREAD_NUM()
!$ size    = OMP_GET_NUM_THREADS()
   IF (my_rank == 0) THEN
     IF (size > 1) THEN
       WRITE(*,'(A,I3,A)') &
        & 'Successful first OpenMP test executed in parallel on ', size, ' threads.'
     ELSE
       WRITE(*,'(A)') 'Caution: This OpenMP test is executed only on one OpenMP thread, i.e., sequentially!'
     ENDIF
   ENDIF
!$OMP END PARALLEL

ELSE

  WRITE(*,'(A)') 'Caution: Your sourcecode was compiled without switching OpenMP on'

ENDIF

stop
end program hello

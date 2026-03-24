program hello
implicit none
integer i
!$ integer OMP_GET_THREAD_NUM

   i=-1

!$OMP PARALLEL

!$ i=OMP_GET_THREAD_NUM()
   print *,'hello world',i

!$OMP END PARALLEL

end program hello

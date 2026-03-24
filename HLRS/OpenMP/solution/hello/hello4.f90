program hello
implicit none
integer i
!$ integer OMP_GET_THREAD_NUM

   i=-1

!$OMP PARALLEL

!$ i=OMP_GET_THREAD_NUM()
   call sleep(1)
   print *,'hello world',i

!$OMP END PARALLEL

end program hello

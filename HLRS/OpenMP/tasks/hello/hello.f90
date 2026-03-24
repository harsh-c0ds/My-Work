program hello
USE omp_lib
implicit none
integer :: itd,i

   !$omp parallel private(itd) 
   itd = omp_get_thread_num()
   i = omp_get_num_threads()
   print *,'hello world',itd, 'of', i
   !$omp end parallel
end program hello

#include <stdio.h>
# ifdef _OPENMP
#  include <omp.h>
# endif

int main(int argc, char** argv)
{
  int my_rank, size;

# ifdef _OPENMP
  #pragma omp parallel private(my_rank, size)
  {
    my_rank = omp_get_thread_num();
    size    = omp_get_num_threads();
    if (my_rank == 0)
    { if (size > 1)
      {
        printf ("Successful first OpenMP test executed in parallel on %i threads.\n", size);
      } else
      {
        printf ("Caution: This OpenMP test is executed only on one OpenMP thread, i.e., sequentially!\n");
      }
    }
  } //end pragma omp
# else
  printf ("Caution: Your sourcecode was compiled without switching OpenMP on\n");
  
# endif
  return 0;
}

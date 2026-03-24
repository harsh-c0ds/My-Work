#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#ifdef _OPENMP
#  include <omp.h>
#endif

#define f(A) (4.0/(1.0+A*A))

const int n = 10000000;

int main(int argc, char** argv)
{
  int i;
  double w,x,sum,pi;
  clock_t t1,t2;
  struct timeval tv1,tv2; struct timezone tz;
# ifdef _OPENMP
    double wt1,wt2;
# endif
 
# ifdef _OPENMP
#   pragma omp parallel
    { 
#     pragma omp single 
      printf("OpenMP-parallel with %1d threads\n", omp_get_num_threads());
    } /* end omp parallel */
# endif

  gettimeofday(&tv1, &tz);
# ifdef _OPENMP
    wt1=omp_get_wtime();
# endif
  t1=clock();

/* calculate pi = integral [0..1] 4/(1+x**2) dx */
  w=1.0/n;
  sum=0.0;
#pragma omp parallel private(x) shared(w)
{
# pragma omp sections
  { 
#  pragma omp section
   {
#   pragma omp taskloop reduction(+:sum) grainsize(n/100)
      /* One can specify "private(x)" on "omp parallel"
         or "omp taskloop", or on both. */ 
     for (i=1;i<=n/100*30;i++)
     {
       x=w*((double)i-0.5);
       sum=sum+f(x);
     }
   } /*end of section */
#  pragma omp section
   {
#   pragma omp taskloop reduction(+:sum) grainsize(n/100)
      /* The number of iterations per task is the same in both loops
         because both loops have the same grain size of n/100.
         So, the first loop, which has 3 million iterations, is
         is split into 30 tasks, while the second loop, which
         has 7 million iterations, is split into 70 tasks. */
     for (i=n/100*30+1;i<=n;i++)
     {
       x=w*((double)i-0.5);
       sum=sum+f(x);
     }
   } /*end of section */
  } /*end of sections */
} /*end omp parallel*/ 
  pi=w*sum;

  t2=clock();
# ifdef _OPENMP
    wt2=omp_get_wtime();
# endif
  gettimeofday(&tv2, &tz);
  printf( "computed pi = %24.16g\n", pi );
  printf( "CPU time (clock)                = %12.4g sec\n", (t2-t1)/1000000.0 );
# ifdef _OPENMP
    printf( "wall clock time (omp_get_wtime) = %12.4g sec\n", wt2-wt1 );
# endif
  printf( "wall clock time (gettimeofday)  = %12.4g sec\n", (tv2.tv_sec-tv1.tv_sec) + (tv2.tv_usec-tv1.tv_usec)*1e-6 );
  return 0;
}

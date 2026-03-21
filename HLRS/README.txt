A) Testing MPI:
---------------

Files are mpitest.c, mpitest_30.f90 and mpitest.py.

typically tested with
  module load .... (to load your compile and MPI environment)
  e.g., module load gnu openmpi
compiled with
  mpicc  mpitest.c
  mpiicc mpitest.c  (with Intel MPI + compiler)
or
  mpif90   mpitest_30.f90
  mpiifort mpitest_30.f90 (with Intel MPI + compiler)

and executed with
  mpirun -np 4 ./a.out                 (or whatever is needed on your system)
or
  mpirun -np 4 --oversubscribe ./a.out (e.g., with OpenMPI and too few cores)
or
  mpirun -np 4 python3 mpitest.py      (python3 version, --oversubscribe might also be necessary)

The expected output should be
  "Successful first MPI test executed in parallel on 4 processes."
  "Your installation supports MPI standard version 3.1."

If the reported number ("... 4 processes") is not identical
to the requested number ("... -np 4 ...") then your installation
is broken !!!

The reported MPI standard version should be 3.1 (or higher in the future) !!!
Otherwise, you should install an recent MPI installation. 

Please test also with 8 and 12 processes (this is required in some exersises).

For the course, it is not relevant whether these 4-12 MPI processes
are running on a cluster or a single ccNUMA system or a single CPU,
and whether each MPI process runs on a single core or whether more
than one (are all) MPI processes run in time-sharing on a core.
 
If you run this test only with one MPI process, then the output will be
  "Caution: This MPI test is executed only on one MPI process, i.e., sequentially!"

This is okay, if you chose "-np 1".

Remark on Fortran:
------------------

All exercises require an installation with the modern mpi_f08 module.
If you can compile and run the C version sucessfully, but the
Fortran version does not compile due to a missing mpi_f08 module,
then there can be several reasons:

- Your installation is without Fortran, then there should not be 
  such a compile script like mpif90 or mpiifort

- The mpi_f08 module was not installed, e.g., due to a wrong default with mpich.
  Then you may use OpenMPI if it is already preinstalled on your system.
  (This happened on my SUSE installationi or with Debian.)

This problem must be resolved prior to the course.
Otherwise, a successful participation is not possible.

For MacOS:
----------

For MacOS, if you want to run the exercises directly on your Mac, 
I received the following hint (not tested by myself):

On macOS can you please:
1. Install Homebrew - open a new terminal and run the command as written on https://brew.sh/
2. Install Open MPI: $ brew install open-mpi
3. Check whether open-mpi and gcc where installed: $ brew list open-mpi gcc
   you should notice open-mpi v4.0.4 and gcc v10
4. Test mpirun version running
        $ mpirun --version
   I would expect somethig like:
        mpirun (Open MPI) 4.0.4
5. It may be necessary that you replace gcc and gfortran in the advices above with:
        - gcc -> gcc-10
        - gfortran -> gfortran-10
   e.g.
        $ gcc-10 -fopenmp openmp-test.c

And please be sure that the installation still works correctly after a reboot of your system.


B) Testing OpenMP:
------------------

Files are openmp-test.c and openmp-test.f90

typically tested with
  module load .... (to load your compiler environment)
  e.g., module load gnu  or   module load intel
compiled with
  gcc -fopenmp openmp-test.c  (with GNU compiler)
  icc -qopenmp openmp-test.c  (with Intel compiler)
or
  gfortran -fopenmp openmp-test.f90 (with GNU compiler)
  ifort    -qopenmp openmp-test.f90 (with Intel compiler)
and executed with
  export OMP_NUM_THREADS=4 ; ./a.out     (on bash, ...)
or 
  setenv OMP_NUM_THREADS 4 ; ./a.out     (on tcsh, ...)

The expected output should be
  "Successful first OpenMP test executed in parallel on 4 threads."

If the reported number ("... 4 threads") is not identical
to the requested number (value of OMP_NUM_THREADS) 
then your installation is broken !!!

Please test also with 8 and 12 processes (this is required in some exersises).

For the course, it is not relevant whether 
each OpenMP thread runs on a single core or whether more
than one (are all) OpenMP threads run in time-sharing on a core.
 
If you run this test only with one OpenMP thread, then the output will be
  "Caution: This OpenMP test is executed only on one OpenMP thread, i.e., sequentially!"

This is okay, if you chose OMP_NUM_THREADS=1.

If you compile without OpenMP support, then the output will be
  "Caution: Your sourcecode was compiled without switching OpenMP on"

In this case, please check for your correct OpenMP compiler flag!

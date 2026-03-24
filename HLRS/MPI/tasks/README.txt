The files in the subdirectories have been written 
as sample solutions to an exercise in a course given 
at the High Performance Computing Centre Stuttgart (HLRS).
The examples in Ch1-Ch13 are based on the examples in the
MPI course of the Edinburgh Parallel Computing Centre (EPCC).
The examples in the other directories are added by HLRS.
It is made freely available with the understanding that
every copy of the files must include their header and that
HLRS and EPCC take no responsibility for the use of the
enclosed teaching material.

Authors: Joel Malard, Alan Simpson,            (EPCC)
         Rolf Rabenseifner, Traugott Streicher (HLRS)

Contact: rabenseifner@hlrs.de

Naming of the files and subdirectories

- MPI/tasks/F_30/*/*skel_30.f90
  MPI/tasks/F_30/*/solutions/*_30.f90

      Examples require MPI-3.0 and later.
      In Fortran, "USE mpi_f08" is used.

      There only a few version-test programs ending with
       _20.f90 using the Fortran mpi module
       _11.f   using the mpif.h include file

- MPI/tasks/C/*/*skel.c
  MPI/tasks/C/*/solutions/*.c
      Valid since MPI-1.1. 
      Should not contain calls to MPI routines
      that are deprecated or removed in a later
      version of MPI.

The examples usually consist of a skeleton in
 

Special directories or links:

- MPI/tasks/C/halo-benchmarks
  MPI/tasks/F_30/halo-benchmarks

      A set of benchmark programs and
      related exercise solutions.

- MPI/tasks/C/eurompi18 as symblic link to
  MPI/tasks/C/Ch9/MPIX/

      Software and files for course chapter 9-(3)

- MPI/tasks/TEST

      Four test files that should be compiled 
      and executed in parallel after you extracted
      one of the tar or zip files:
        MPI31.tar.gz, MPI31single.tar.gz, MPI31.zip
      from 
        https://fs.hlrs.de/projects/par/par_prog_ws/practical/README.html

      For further instructions, see
        MPI/tasks/TEST/README.txt

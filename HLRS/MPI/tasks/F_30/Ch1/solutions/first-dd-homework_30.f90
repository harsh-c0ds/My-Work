PROGRAM first_dd_homework

!==============================================================!
!                                                              !
! This file has been written as a sample solution to an        !
! exercise in a course given at the High Performance           !
! Computing Centre Stuttgart (HLRS).                           !
! The examples are based on the examples in the MPI course of  !
! the Edinburgh Parallel Computing Centre (EPCC).              !
! It is made freely available with the understanding that      !
! every copy of this file must include this header and that    !
! HLRS and EPCC take no responsibility for the use of the      !
! enclosed teaching material.                                  !
!                                                              !
! Authors: Joel Malard, Alan Simpson,            (EPCC)        !
!          Rolf Rabenseifner, Traugott Streicher (HLRS)        !
!                                                              !
! Contact: rabenseifner@hlrs.de                                !
!                                                              !
! Purpose: A second MPI example calculating the subdomain size !
!                                                              !
! Contents: F-Source                                           !
!                                                              !
!==============================================================!

  USE mpi_f08

  IMPLICIT NONE

  integer :: idim=3  ! = number of processes in this dimension
  integer :: icoord  ! = number of a process in this dimension: between 0 and idim-1
  integer :: istart=0, imax=80  ! = start and last index of the global data mesh
                                !   in one dimension, including the boundary condition.
  integer :: isize      ! = length of the global data mesh in this dimension, i.e.,
                        ! = imax-istart+1
  integer :: b1=1   ! Width of the boundary condition = width of the halos
                    ! in this dimension.
  integer :: is, ie   ! start and end index of the subarray (of the domain decomposition)
                      ! in this dimension in process icoord 
  integer :: iouter   ! = ie-is+1 = size of the local subdomain data mesh in that dimension
  integer :: iinner   ! = iouter-2*b1 = number of unknowns in this dimension of process icoord
  integer :: iinner0  ! = smallest number of unknowns in a process in this dimension
  integer :: in1      ! = number of processes with inner=inner0+1

  integer :: numprocs, my_rank  ! additional variables for parallelization

! Setting: This program should run only with one MPI process and calculates the
!          subarray length  iouter  and its indices from  is  to  ie  (as in the global array)
!          for each of the  idim  processes in one of dimension,
!          i.e., for process coordinate  icoord  between  0  and  idim-1 
! Given:   idim, istart, imax, (and therefore isize), b1
! Goal:    to calculate is, ie (and iouter and iinner) for each icoord between 0 and idim-1

  call MPI_Init()
  call MPI_Comm_rank(MPI_COMM_WORLD, my_rank)
  call MPI_Comm_size(MPI_COMM_WORLD, numprocs)

if(my_rank==0) then
 ! run this test only with one MPI process

  write (*,'(/,a)') 'Type in imax (=last index of the global data mesh) and'
  write (*,'(a)')   'idim (=number of processes in a dimension) and'
  write (*,'(a,i2,x,i2,x,i2)') 'b1 (=width of boundary = halo width), e.g. ',imax,idim,b1

  read (*,*) imax, idim, b1


  write (*,'(/,a,i3,a,i3)') 'indices of the global data mesh are between',istart,' and ',imax
  write (*,'(a,i3)')        'number of processes in this dimension = ',idim
  write (*,'(a,i3)')        'boundary width = halo width is ',b1
  write (*,'(a,i3,a,i3)')   'indices of the unknowns are between ',istart+b1,' and ',imax-b1

 do icoord=0,idim-1
  ! emulating all processes in one dimension 
  
  
! Calculating the own subdomain in each process
! ---------------------------------------------
!
! whole y indices    |------------- isize = imax-istart+1 ------------|
!    start/end index  ^-istart                                  imax-^
! 
! 1. interval        |--- iouter1---|   
!                    |--|--------|--|
!                     b1  iinner1 b1 
!    start/end index  ^-is      ie-^ 
! 2. interval                 |--|--------|--|   
! 3. interval                          |--|--------|--| 
! 4. interval                                   |--|-------|--| 
!                                                   iinner0 = iinner1 - 1
! 5. interval = idim's interval                         |--|-------|--|
!
! In each iteration on each interval, the inner area is computed
! by using the values of the last iteration in the whole outer area. 
!
! icoord = number of the interval - 1
! 
! To fit exactly into isize, we use in1 intervals of with iinner1 = iinner0 + 1
! and (idim-in1) intervals of with iinner0 
!
!         Originally:     And as result of the domain decomposition into idim subdomains:
! isize = imax-istart+1 = 2*b1 + in1*iinner1 + (idim-in1)*inner0
!
! computing is:ie, ks:ke
!   - input:            istart, imax, b1, idim, icoord (and k...)
!   - to be calculated: is, ie, iinner, iouter
!   - helper variables: iinner0, in1, isize

  isize = imax - istart + 1  ! total number of elements, including the "2*b1" boundary elements
   ! isize - 2*b1 = total number of unknowns
  iinner0 = (isize - 2*b1)  / idim  ! smaller inner size through divide with rounding off
  in1 = isize - 2*b1 - idim * iinner0  ! number of processes that must have "inner0+1" unknowns
  if (icoord < in1) then   ! the first in1 processes will have "iinner0+1" unknowns
    iinner = iinner0 + 1
    is = (istart+b1) + icoord * iinner - b1  ! note that "is" reflects the position of the 
                                             ! first halo or boundary element of the subdomain
  else                     ! and all other processes will have iinner0 unknowns
    iinner = iinner0
    is = istart + in1 * (iinner0+1) + (icoord-in1) * iinner
  endif
  iouter = iinner + 2*b1
  ie = is + iouter - 1

  if(icoord==0)then
    write (*,'(/,a)') 'Please control whether isize and the sum are identical:'
    write (*,'(/,2(1x,a,i3),2(1x,a,i3,1x,a,i3),2(1x,a,i3))') &
     &           'isize=',isize,'idim=',idim, &
     &                         '|| in1*(iinner0+1)=',in1,'*',iinner0+1,  '+ (idim-in1)*iinner0=',idim-in1,'*',iinner0, &
     &                         '+ 2*b1=2*',b1,'|| sum = ',in1*(iinner0+1) + (idim-in1)*iinner0+2*b1
    write (*,'(/,a,i3,a,i3,a)') 'Please control whether the indices of unkowns are between ', &
     &                            istart+b1, ' ..', imax-b1, ', complete, and non-overlapping:'
  endif

  if(iinner>0) then
    write (*,'(7(a,i3))')   ' icoord=',icoord,', iouter=',iouter,', iinner=',iinner, &
     &                      ', subarray indices= ',is,' ..',ie,', indices of the unknowns= ',is+b1,' ..',ie-b1
  else
    write (*,'(3(a,i3),a)') ' icoord=',icoord,', iouter=',iouter,', iinner=',iinner,', no subarray'
  endif


 end do !icoord=0,idim-1

endif !(my_rank==0)

  call MPI_Finalize()

END PROGRAM first_dd_homework

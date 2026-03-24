PROGRAM ring

!**************************************************************!
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
! Purpose: A program to try MPI_Issend and MPI_Recv.           !
!                                                              !
! Contents: C-Source                                           !
!                                                              !
!**************************************************************!


  USE mpi_f08

  IMPLICIT NONE

  INTEGER, PARAMETER :: to_right=201

  INTEGER :: my_world_rank, world_size, my_sub_rank, sub_size
  INTEGER :: snd_buf, rcv_buf
  INTEGER :: right, left
  INTEGER :: sumA, sumB, i
  INTEGER :: mycolor
  INTEGER :: ranges(3,1)
  TYPE(MPI_Group) :: world_group, sub_group
  TYPE(MPI_Comm ) :: sub_comm

  TYPE(MPI_Status ) :: status
  TYPE(MPI_Request) :: request


  CALL MPI_Init()

  CALL MPI_Comm_size(MPI_COMM_WORLD, world_size)
  CALL MPI_Comm_rank(MPI_COMM_WORLD, my_world_rank)

  IF (my_world_rank > (world_size-1)/3) THEN
    mycolor = 1
  ELSE
    mycolor = 0
  END IF
  ! This definition of mycolor implies that the first color is 0

  ! instead of  
  !  CALL MPI_Comm_split(MPI_COMM_WORLD, mycolor, 0, sub_comm) 
  !  ... the following code is used: 

  CALL MPI_Comm_group(MPI_COMM_WORLD, world_group)

  IF (mycolor == 0) THEN
    ranges(1,1) = 0                     ! first rank of my range
    ranges(2,1) = (world_size-1)/3      ! last  rank of my range
  ELSE
    ranges(1,1) = (world_size-1)/3 + 1  ! first rank of my range
    ranges(2,1) = world_size-1          ! last  rank of my range
  END IF
  ranges(3,1) = 1                       ! stride of ranks

  ! write(*,'(  ''PE world:'',I3,'', color='',I3,'' first='',I3,'', last='',I3,'', stride='',I3)') &
  ! &             my_world_rank,     mycolor,       ranges(1,1),   ranges(2,1),  ranges(3,1)
  
  CALL MPI_Group_range_incl ( world_group, 1, ranges, sub_group)
  CALL MPI_Comm_create(MPI_COMM_WORLD, sub_group, sub_comm)

  CALL MPI_Comm_size(sub_comm, sub_size)
  CALL MPI_Comm_rank(sub_comm, my_sub_rank)

  ! Compute sum of all ranks.
  CALL MPI_Allreduce (my_world_rank, sumA, 1, MPI_INTEGER, MPI_SUM, sub_comm) 
  CALL MPI_Allreduce (my_sub_rank,   sumB, 1, MPI_INTEGER, MPI_SUM, sub_comm) 

  write(*,'(''PE world:'',I3,'', color='',I3,'' sub:'',I3,'' SumA='',I5,'' SumB='',I5,'' in subcomm'')')&
  &           my_world_rank,     mycolor,    my_sub_rank,    sumA,         sumB

  CALL MPI_Finalize()
END PROGRAM

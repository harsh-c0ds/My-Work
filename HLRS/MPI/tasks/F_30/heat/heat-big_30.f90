program heat
  implicit none
! naming: i = 1st array coordinate = 1st process coordinate = y
!         k = 2nd array coordinate = 2nd process coordinate = x
!                                                    Caution: y,x and not x,y
  integer i, k, it

  logical, parameter :: prt=.true.
! logical, parameter :: prt=.false.
  integer, parameter :: imax=80, kmax=80
  integer, parameter :: istart=0, kstart=0, b1=1
  integer, parameter :: itmax=20000
  double precision, parameter :: eps=1.d-08
 
  double precision, allocatable :: phi(:,:), phin(:,:)
  double precision dx,dy,dx2,dy2,dx2i,dy2i,dt,dphi,dphimax

  real telapsed, tarray(2) ! tarray(1) = user time, tarray(2) = system time since (start or) last call

! Preparation for parallelization with domain decomposition:

  integer :: is=istart, ie=imax, ks=kstart, ke=kmax 

! The algortithm below is already formulated with lower and upper 
! index values (is:ie), (ks:ke) instead of (istart:imax), (kstart:kmax).
! With the MPI domain decomposition, (is:ie) and (ks:ke) will define the index rang 
! of the subdomain in a given MPI process.

  allocate(phi(is:ie,ks:ke))
  allocate(phin(is+b1:ie-b1,ks+b1:ke-b1))
!         
! naming: i = 1st array coordinate = 1st process coordinate = y
!         k = 2nd array coordinate = 2nd process coordinate = x
!                                                    Caution: y,x and not x,y
  dx=1.d0/(kmax-kstart)
  dy=1.d0/(imax-istart)
  dx2=dx*dx
  dy2=dy*dy
  dx2i=1.d0/dx2
  dy2i=1.d0/dy2
  dt=min(dx2,dy2)/4.d0
! start values 0.d0 
  do k=ks, min(ke,kmax-1)
    do i=max(1,is), min(ie,imax-1) ! do not overwrite the boundary condition
      phi(i,k)=0.d0
    enddo
  enddo
! start values 1.d0
 if (ke.eq.kmax) then 
  do i=is,ie
    phi(i,kmax)=1.d0
  enddo
 endif 
! start values dx
 if (is.eq.istart) then 
!       phi(0,0)=0.d0
!       do k=1,kmax-1 
!         phi(0,k)=phi(0,k-1)+dx 
!         phi(imax,k)=phi(imax,k-1)+dx 
!       enddo 
!       ... substitute algorithmus by a code, 
!           that can be computed locally: 
  do k=ks,min(ke,kmax-1)
    phi(istart,k) = (k-kstart)*dx 
  enddo
 endif 
 if (ie.eq.imax) then 
  do k=ks,min(ke,kmax-1)
    phi(imax,k) = (k-kstart)*dx 
  enddo
 endif 
 
! print details
       write (*,'(/,a)') 'Heat Conduction 2d'
       write (*,'(/,4(a,1pg12.4))') 'dx =',dx,', dy =',dy,', dt =',dt,', eps =',eps

call etime(tarray, telapsed)
 
! iteration
  do it=1,itmax
      dphimax=0.
      do k=ks+b1,ke-b1
       do i=is+b1,ie-b1
        dphi=(phi(i+1,k)+phi(i-1,k)-2.*phi(i,k))*dy2i &
 &             +(phi(i,k+1)+phi(i,k-1)-2.*phi(i,k))*dx2i
        dphi=dphi*dt
        dphimax=max(dphimax,dphi)
        phin(i,k)=phi(i,k)+dphi
       enddo
      enddo
! save values
      do k=ks+b1,ke-b1
       do i=is+b1,ie-b1
        phi(i,k)=phin(i,k)
       enddo
      enddo
 
      if(dphimax.lt.eps) then
          goto 10 ! Finish the timestep loop "do it=…"
      endif 
 
  enddo
10      continue

call etime(tarray, telapsed)

  if (prt) then
            write(*,'(''     i='',1000(1x,i4))') &
                                  (i,i=is,ie) 
          do k=ks,ke
            write(*,'('' k='',i3,'':'',1000(1x,f4.2))') &
                              k,       (phi(i,k),i=is,ie)
          enddo
  endif 

   write (*,'(/a/a/)') &
 &'! iter-     elapsed time  user time    system time', &
 &'! ations    [seconds]     [seconds]     [seconds]   '
   write(*, &
 &   '(''!'',i6,2x,g12.4,2x,g12.4,2x,g12.4)') &
 &           it, telapsed, tarray(1), tarray(2)

end program heat

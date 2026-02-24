!  Integration_methods.f90 
! Methods:
! - Simpson's 3/8 rule
! - Gauss_Legendre methods of order 2, 4, and 8
!****************************************************************************

    program Integration_methods
        implicit none

        integer, parameter :: kk = SELECTED_REAL_KIND(16, 200)
        real(kind = kk), dimension(3) :: res_1, res_2, res_3, res_4, res_5, res_6
        real(kind = kk), dimension(497) :: error1, log_h1, error2, log_h2, error3, log_h3, error4, log_h4, error5, log_h5, error6, log_h6
        real(kind = kk), dimension(2) :: lim_int_1=(/0., 3.141592653589793/2./), lim_int_2=(/-1., 3./), lim_int_3=(/-1., 1./)  ! limits of integration
        integer :: n, m, counter = 0
        character(len=30) :: name1 = "doc_1.txt"
        character(len=30) :: name2 = "doc_2.txt"
        character(len=30) :: name3 = "doc_3.txt"
        character(len=30) :: name4 = "doc_4.txt"
        character(len=30) :: name5 = "doc_5.txt"
        character(len=30) :: name6 = "doc_6.txt"
        real(kind = kk) :: exact1 = 1.90523869, exact2 = 19.71765748, exact3 = 3.24939038
        
        do n = 5, 501
            m = n - 1  ! number of intervals
            counter = counter + 1
            
            res_1 = simpson3_8(lim_int_1, m, f1, error1_3, error3_8, error_trap)
            error1(counter) = res_1(2)
            log_h1(counter) = log(res_1(3))
            
            res_2 = simpson3_8(lim_int_2, m, f2, error1_3, error3_8, error_trap)
            error2(counter) = res_2(2)
            log_h2(counter) = log(res_2(3))
            
            res_3 = simpson3_8(lim_int_3, m, f3, error1_3, error3_8, error_trap)
            error3(counter) = res_3(2)
            log_h3(counter) = log(res_3(3))  
        end do
        print*, "Simpson's 3/8 rule"
        print*, "f1", res_1(1)
        print*, "f2", res_2(1)
        print*, "f3", res_3(1)
        print*, "" 
        
        call make_doc(log_h1, error1, name1)
        call make_doc(log_h2, error2, name2)
        call make_doc(log_h3, error3, name3)
        
        counter = 0
        do n = 5, 501
            m = n - 1  ! number of intervals
            counter = counter + 1
            
            res_4 = gauss_leg(lim_int_1, m, f1, exact1)
            error4(counter) = res_4(2)
            log_h4(counter) = res_4(3)
            
            res_5 = gauss_leg(lim_int_2, m, f2, exact2)
            error5(counter) = res_5(2)
            log_h5(counter) = res_5(3)
            
            res_6 = gauss_leg(lim_int_3, m, f3, exact3)
            error6(counter) = res_6(2)
            log_h6(counter) = res_6(3)
        end do
        print*, "Gauss-Legendre quadrature"
        print*, "f1", res_4(1)
        print*, "f2", res_5(1)
        print*, "f3", res_6(1)
    
        call make_doc(log_h4, error4, name4)
        call make_doc(log_h5, error5, name5)
        call make_doc(log_h6, error6, name6)
        
        
    contains
    
    ! THREE FUNCTIONS, f1, f2, and f3, TO BE INTEGRATED
        real(kind = SELECTED_REAL_KIND(16, 200)) function f1(x)
            implicit none
            
            integer, parameter :: kk = SELECTED_REAL_KIND(16, 200)
            real(kind = kk) :: x
            
            f1 = EXP(x) * COS(x)
        end function f1
        
        
        real(kind = SELECTED_REAL_KIND(16, 200)) function f2(x)
            implicit none
            
            integer, parameter :: kk = SELECTED_REAL_KIND(16, 200)
            real(kind = kk) :: x
            
            f2 = EXP(x)
        end function f2
        
        
        real(kind = SELECTED_REAL_KIND(16, 200)) function f3(x)
            implicit none
            
            integer, parameter :: kk = SELECTED_REAL_KIND(16, 200)
            real(kind = kk) :: x
            
            if (x < 0) then
                f3 = EXP(2*x) 
            else
                f3 = x - 2*COS(x) + 4
            end if
        end function f3
            
    
    ! ****************************************************************************************
    ! SIMPSON'S 3/8 RULE 
    ! ****************************************************************************************
        function simpson3_8(lim_int, m, func, error1_3, error3_8, error_trap) result(output)
            implicit none
            
            integer, parameter :: kk = SELECTED_REAL_KIND(16, 200)
            real(kind = kk) :: h, res, error3_8, error1_3, error_trap
            real(kind = kk), allocatable :: x_vec(:)  ! x_0, ..., x_n
            real(kind = kk), dimension(2) :: lim_int, lim_1, lim_2
            real(kind = kk), dimension(3) :: output
            integer :: m, n, i, temp
            real(kind = kk), external :: func
            
            n = m + 1
            
            allocate(x_vec(n))
            
            h = (lim_int(2)-lim_int(1)) / m
            output(3) = h
            
            do i = 0, m
                x_vec(i+1) = lim_int(1) + i*h
            end do
            
            res = 0.
            
            if (mod(m, 3) == 0) then
                !print*, "mod=0"
                ! Simpson's 3/8 rule
                do i = 1, n
                    if (i == 1 .OR. i == n) then
                        res = res + func(x_vec(i))
                    elseif (mod(i-1, 3) == 0) then
                        res = res + 2 * func(x_vec(i))
                    else
                        res = res + 3 * func(x_vec(i))
                    end if
                end do
                output(1) = 3./8. * h * res
                output(2) = error3_8(lim_int, m, func)

            elseif (mod(m,3) == 2) then
                !print*, "mod=2"
                temp = n - 3
                ! Simpson's 3/8 rule
                do i = 1, temp
                    if (i == 1 .OR. i == temp) then
                        res = res + func(x_vec(i))
                    elseif (mod(i-1, 3) == 0) then
                        res = res + 2 * func(x_vec(i))
                    else
                        res = res + 3 * func(x_vec(i))
                    end if
                end do
                res = 3./8. * h * res
                ! Simpson's 1/3 rule for the last 2 intervals
                output(1) = res + 1./3. * h * (func(x_vec(n-2)) + 4*func(x_vec(n-1)) + func(x_vec(n)))
                lim_1 = (/x_vec(1), x_vec(temp)/)
                lim_2 = (/x_vec(temp+1), x_vec(n)/)
                output(2) = error3_8(lim_1, m, func) + error1_3(lim_2, m, func)
                
            elseif (mod(m,3) == 1) then
                !print*, "mod=1"
                temp = n - 2
                ! Simpson's 3/8 rule
                do i = 1, temp
                    if (i == 1 .OR. i == temp) then
                        res = res + func(x_vec(i))
                    elseif (mod(i-1, 3) == 0) then
                        res = res + 2 * func(x_vec(i))
                    else
                        res = res + 3 * func(x_vec(i))
                    end if
                end do
                res = 3./8. * h * res
                ! Trapezoidal rule for the last interval
                output(1) = res + h/2. * (func(x_vec(n-1)) + func(x_vec(n)))
                lim_1 = (/x_vec(1), x_vec(temp)/)
                lim_2 = (/x_vec(temp+1), x_vec(n)/)
                output(2) = error3_8(lim_1, m, func) + error_trap(lim_2, m, func)
            end if    
        deallocate(x_vec)       
        end function simpson3_8
    
        
    ! ****************************************************************************************
    ! GAUSS-LEGENDRE QUADRATURE (SECOND ORDER)
    ! ****************************************************************************************
        function gauss_leg(lim_int, m, func, value) result(output)
            implicit none
            
            real(kind = kk) :: h, func, res
            REAL(KIND = kk) :: value
            real(kind = kk), allocatable :: x_vec(:)  ! x_0, ..., x_n
            real(KIND = kk), dimension(2) :: lim_int, xi = (/-0.57735, 0.57735/)  ! Nodes of Legendre polynomial
            real(KIND = kk), dimension(3) :: output
            integer :: m, n, i, j
                
            n = m + 1
            
            allocate(x_vec(n))
            
            h = (lim_int(2)-lim_int(1)) / m
            
            do i = 0, m
                x_vec(i+1) = lim_int(1) + i*h
            end do
            
            res = 0.
            
            ! Gauss-Legendre quadrature of second order
            do i = 1, n - 1
                do j = 1, 2
                    res = res + h/2. * func(h/2. * xi(j) + (x_vec(i+1) + x_vec(i)) / 2.)
                end do
            end do
            output(1) = res
            output(2) = res - value
            !print*, "res-value", output(2)
            output(3) = LOG(h)
            
            deallocate(x_vec)
        end function gauss_leg
        
        
    ! *************************************************************************************    
    ! FUNCTIONS FOR CALCULATING ERRORS
    ! *************************************************************************************
        real(kind = SELECTED_REAL_KIND(16, 200)) function error1_3(lim_int, m, func) 
            implicit none
            
            real(kind = kk) :: func, fourth_deriv, x_0, h
            real(kind = kk), allocatable :: x_vec(:)
            integer :: m, n, i, index_0
            real(KIND = kk), dimension(2) :: lim_int
            
            n = m + 1
            allocate(x_vec(n))
            
            h = (lim_int(2)-lim_int(1)) / m
            
            do i = 0, m
                x_vec(i+1) = lim_int(1) + i*h
            end do
            
            if (n<30) then
                index_0 = ANINT(n/2.)  ! I choose the central point to calculate the derivative
            else
                index_0 = ANINT(n/2.-10) 
            end if
            x_0 = x_vec(index_0)
            
            fourth_deriv = (func(x_vec(index_0+2)) - 4*func(x_vec(index_0+1)) + 6*x_0 - 4*func(x_vec(index_0-1)) + func(x_vec(index_0-2))) / (h**4)
        
            error1_3 = -(lim_int(2)-lim_int(1)) / 180. * h**4 * fourth_deriv
            
            deallocate(x_vec)
        end function error1_3
        
        
        real(kind = SELECTED_REAL_KIND(16, 200)) function error3_8(lim_int, m, func) 
            implicit none
            
            real(kind = kk) :: func, fourth_deriv, x_0, h
            real(kind = kk), allocatable :: x_vec(:)
            integer :: m, n, i, index_0
            real(KIND = kk), dimension(2) :: lim_int
            
            n = m + 1
            allocate(x_vec(n))
            
            h = (lim_int(2)-lim_int(1)) / m
            
            do i = 0, m
                x_vec(i+1) = lim_int(1) + i*h
            end do
            
            if (n<30) then
                index_0 = ANINT(n/2.)  ! I choose the central point to calculate the derivative
            else
                index_0 = ANINT(n/2.-10) 
            end if
            x_0 = x_vec(index_0)
            
            fourth_deriv = (func(x_vec(index_0+2)) - 4*func(x_vec(index_0+1)) + 6*x_0 - 4*func(x_vec(index_0-1)) + func(x_vec(index_0-2))) / (h**4)
        
            error3_8 = -(lim_int(2)-lim_int(1)) / 80. * h**4 * fourth_deriv
            
            deallocate(x_vec)
        end function error3_8
        
        
        real(kind = SELECTED_REAL_KIND(16, 200)) function error_trap(lim_int, m, func) 
            implicit none
            
            real(kind = kk) :: func, second_deriv, x_0, h
            real(kind = kk), allocatable :: x_vec(:)
            integer :: m, n, i, index_0
            real(KIND = kk), dimension(2) :: lim_int
            
            n = m + 1
            allocate(x_vec(n))
            
            h = (lim_int(2)-lim_int(1)) / m
            
            do i = 0, m
                x_vec(i+1) = lim_int(1) + i*h
            end do
            
            if (n<30) then
                index_0 = ANINT(n/2.)  ! I choose the central point to calculate the derivative
            else
                index_0 = ANINT(n/2.-10) 
            end if
            x_0 = x_vec(index_0)
            
            second_deriv = (-func(x_vec(index_0+2)) + 16*func(x_vec(index_0+1)) - 30*x_0 + 16*func(x_vec(index_0-1)) - func(x_vec(index_0-2))) / (12*h**2)
        
            error_trap = -(lim_int(2)-lim_int(1)) / 12. * h**2 * second_deriv
            
            deallocate(x_vec)
        end function error_trap
        
        
        subroutine make_doc(x, y, name)

            integer, parameter :: kk = SELECTED_REAL_KIND(16, 200)
            real(KIND = kk), dimension(497) :: x, y
            integer :: i
            character(len=30) :: name
            
            open(unit=10, file=name, status='replace', action='write', iostat=i)

            ! Write the data to the file
            do i = 1, size(x)
                write(10, '(E30.20, 1X, E30.20)') x(i), y(i)
            end do

            close(10)
            print *, "Data written to", trim(name)
        end subroutine make_doc

            
    end program Integration_methods
program matmul
    implicit none

    integer, parameter :: n = 10
    real(8), dimension(n,n) :: a, b, c
    integer :: i, j

    ! Initialize values of input matrices
    do i = 1, n
        do j = 1, n
            call random_number(a(i,j))
            call random_number(b(i,j))
        end do
    end do

    call dgemm("N", "N", n, n, n, 1.0d0, a, n, b, n, 0.0d0, c, n)

    write (*,*) "C matrix Frobenius norm = ", norm2(c)
end program

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

    write (*,*) "C matrix Frobenius norm = ", frob_norm(c)

contains
    function frob_norm(input)
        real(8), intent(in) :: input(:,:)
        real(8) :: frob_norm
        integer :: i, j

        frob_norm = 0.0d0

        do i = 1, size(input, 1)
            do j = 1, size(input, 2)
                frob_norm = frob_norm + input(i,j)**2.0d0
            end do
        end do

        frob_norm = sqrt(frob_norm)
    end function
end program

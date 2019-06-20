program matmul
    implicit none

    integer, parameter :: n = 10
    real(8), dimension(n) :: a, b, c
    integer :: i, j

    ! Initialize values of input matrices
    do i = 1, n
        do j = 1, n
            call random_number(a(i,j))
            call random_number(b(i,j))
        end do
    end do

end program

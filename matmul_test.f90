program matmul_test
    use cublas_gemm_f, only: tcgemm

    implicit none

    ! Size of matrices
    integer, parameter :: n = 10

    ! Host matrices
    real(8), dimension(n,n) :: a, b, c

    integer :: i, j

    ! Initialize values of input matrices
    do i = 1, n
        do j = 1, n
            call random_number(a(i,j))
            call random_number(b(i,j))
        end do
    end do

    ! =========================================================================
    ! Host DGEMM
    ! =========================================================================

    c = matmul(a, b)

    write (*,"(A35,F13.10)") "C matrix Frobenius norm (host)   = ", frob_norm(c)

    ! =========================================================================
    ! Device DGEMM
    ! =========================================================================

    ! Call Tensor Core GEMM routine
    call tcgemm("N", "N", n, n, n, 1.0d0, a, n, b, n, 0.0d0, c, n)

    write (*,"(A35,F13.10)") "C matrix Frobenius norm (device) = ", frob_norm(c)

contains
    ! Computes Frobenius norm of input matrix
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

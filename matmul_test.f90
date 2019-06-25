program matmul_test
    use cublas_gemm_f, only: tcgemm

    implicit none

    ! Size of matrices
    integer, parameter :: n = 10

    ! Host matrices
    real(8), dimension(n,n) :: a1, b1, c1, a2, b2
    real(4), dimension(n,n) :: c2

    integer :: i, j

    ! Initialize values of input matrices
    do i = 1, n
        do j = 1, n
            call random_number(a1(i,j))
            call random_number(b1(i,j))
            a2(i,j) = a1(i,j)
            b2(i,j) = b1(i,j)
        end do
    end do

    ! =========================================================================
    ! Host DGEMM
    ! =========================================================================

    c1 = matmul(a1, b1)

    write (*,"(A35,F13.10)") "C matrix Frobenius norm (host)   = ", frob_norm(c1)

    ! =========================================================================
    ! Device DGEMM
    ! =========================================================================

    ! Call Tensor Core GEMM routine
    call tcgemm("N", "N", n, n, n, 1.0, a2, n, b2, n, 0.0, c2, n)

    write (*,"(A35,F13.10)") "C matrix Frobenius norm (device) = ", frob_norm(real(c2,8))

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

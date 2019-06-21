program matmul_test
    implicit none

    ! Size of matrices
    integer, parameter :: n = 10

    ! Host matrices
    real(8), dimension(n,n) :: a, b, c

    ! Pointers to device matrices
    integer(8) :: a_d, b_d, c_d

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

    ! Allocate memory on device and transfer input data from host
    call cublas_alloc(n*n, 8, a_d)
    call cublas_alloc(n*n, 8, b_d)
    call cublas_alloc(n*n, 8, c_d)
    call cublas_set_matrix(n, n, 8, a, n, a_d, n)
    call cublas_set_matrix(n, n, 8, b, n, b_d, n)

    ! Call DGEMM routine
    call cublas_dgemm("N", "N", n, n, n, 1.0d0, a_d, n, b_d, n, 0.0d0, c_d, n)

    ! Transfer results back to host
    call cublas_get_matrix(n, n, 8, c_d, n, c, n)

    write (*,"(A35,F13.10)") "C matrix Frobenius norm (device) = ", frob_norm(c)

    ! Free GPU memory
    call cublas_free(a_d)
    call cublas_free(b_d)
    call cublas_free(c_d)

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

program matmul_test
    use cublas_gemm_f, only: init_gpu, fin_gpu, tcgemm

    implicit none

    ! Size of matrices
    integer, parameter :: m = 10, n = 15

    ! Host matrices
    real(8), dimension(m,n) :: a1, b1, a2, b2
    real(8), dimension(m,m) :: c1
    real(4), dimension(m,m) :: c2
    real(4) :: tick, tock

    integer :: i, j

    ! Initialize values of input matrices
    do i = 1, m
        do j = 1, n
            call random_number(a1(i,j))
            call random_number(b1(i,j))
            a2(i,j) = a1(i,j)
            b2(i,j) = b1(i,j)
        end do
    end do

    ! =========================================================================
    ! Host DGEMM (with transpose)
    ! =========================================================================

    call cpu_time(tick)
    call dgemm("N", "T", m, m, n, 1.0d0, a1, m, b1, m, 0.0d0, c1, m)
    call cpu_time(tock)

    write (*,"(A35,F17.10)") "C matrix Frobenius norm (host)   = ", frob_norm(c1)
    write (*,"(A11,F13.10)") "CPU time = ", tock - tick

    ! =========================================================================
    ! Device DGEMM (with transpose)
    ! =========================================================================

    call init_gpu(m, m, n)

    ! Call Tensor Core GEMM routine
    call cpu_time(tick)
    call tcgemm("N", "T", m, m, n, 1.0, a2, m, b2, m, 0.0, c2, m)
    call cpu_time(tock)

    call fin_gpu

    write (*,"(A35,F17.10)") "C matrix Frobenius norm (device) = ", frob_norm(real(c2,8))
    write (*,"(A11,F13.10)") "GPU time = ", tock - tick

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

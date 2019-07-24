!> Provides wrappers for performing matrix-matrix multiplication on a GPU using
!  CUBLAS.
 module cublas_gemm_f
    implicit none

    interface
        !> Perform matrix-matrix multiplication using Tensor Core (interface for C
        !  function).
        subroutine tcgemm_c(transa, transb, m, n, k, alpha, a_p, lda, b_p, ldb, beta, c_p, ldc) &
                & bind(c)
            use iso_c_binding, only: c_char, c_int, c_float, c_ptr
            integer(kind=c_int), value :: transa, transb
            integer(kind=c_int), value :: m, n, k, lda, ldb, ldc
            real(kind=c_float), value :: alpha, beta
            type(c_ptr), value :: a_p, b_p, c_p
        end subroutine
    end interface

contains
    !> Perform matrix-matrix multiplication using Tensor Core (wrapper for C
    !  function).
    subroutine tcgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
        use iso_c_binding, only: c_ptr, c_loc

        character :: transa, transb
        integer :: m, n, k, lda, ldb, ldc
        real(4) :: alpha, beta
        real(8), target :: a(:,:), b(:,:)
        real(4), target :: c(:,:)
        type(c_ptr) :: a_p, b_p, c_p
        integer :: transa_l, transb_l

        ! Copy data to C pointers
        a_p = c_loc(a(1,1))
        b_p = c_loc(b(1,1))
        c_p = c_loc(c(1,1))

        ! TODO: Figure out how to pass single character strings to C
        transa_l = merge(0, 1, transa == "N" .or. transa == "n")
        transb_l = merge(0, 1, transb == "N" .or. transb == "n")

        ! Call C function
        call tcgemm_c(transa_l, transb_l, m, n, k, alpha, a_p, lda, b_p, ldb, beta, c_p, ldc)
    end subroutine
end module

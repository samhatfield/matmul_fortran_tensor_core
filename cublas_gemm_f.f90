!> Provides wrappers for performing matrix-matrix multiplication on a GPU using
!  CUBLAS.
 module cublas_gemm_f
    implicit none

    interface
        !> Perform matrix-matrix multiplication using Tensor Core (interface for C
        !  function).
        subroutine tcgemm_c(transa, transb, m, n, k, alpha, a_p, lda, b_p, ldb, beta, c_p, ldc) &
                & bind(c, name="tcgemm_c")
            use iso_c_binding, only: c_char, c_int, c_double, c_ptr
            character(kind=c_char), value :: transa, transb
            integer(kind=c_int), value :: m, n, k, lda, ldb, ldc
            real(kind=c_double), value :: alpha, beta
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
        real(8) :: alpha, beta
        real(8), target :: a(:,:), b(:,:), c(:,:)
        type(c_ptr) :: a_p, b_p, c_p

        ! Copy data to C pointers
        a_p = c_loc(a(1,1))
        b_p = c_loc(b(1,1))
        c_p = c_loc(c(1,1))

        ! Call C function
        call tcgemm_c("N", "N", n, n, n, 1.0d0, a_p, n, b_p, n, 0.0d0, c_p, n)
    end subroutine
end module


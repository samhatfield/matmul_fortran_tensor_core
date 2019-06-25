#include <stdio.h>
#include <cublas_v2.h>

// Handles CUDA errors
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

// Handles cuBLAS errors
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

// Performs matrix-matrix multiplication using Tensor Core.
extern "C" {
    void tcgemm_c(char transa, char transb, int m, int n, int k, double alpha, void *a_p, int lda, void *b_p,
                int ldb, double beta, void *c_p, int ldc) {
    
        // Set up host-side arrays
        double *a_h, *b_h, *c_h;
        a_h = (double *)a_p;
        b_h = (double *)b_p;
        c_h = (double *)c_p;
    
        // =========================================================================
        // Compute GEMM using Tensor Core
        // =========================================================================
    
        // Set up GPU and cuBLAS
        cublasHandle_t cublasHandle;
        cudaSetDevice(0);
        cudaDeviceReset();
        cublasErrCheck(cublasCreate(&cublasHandle));
    
        // Set up device-side arrays
        double *a_d, *b_d, *c_d;
    
        // Allocate memory on device for all arrays
        // TODO: should the dimensions used below (m*k etc.) take into account transa, lda etc.?
        cudaErrCheck(cudaMalloc((void **)&a_d, m*k*sizeof(double)));
        cudaErrCheck(cudaMalloc((void **)&b_d, k*n*sizeof(double)));
        cudaErrCheck(cudaMalloc((void **)&c_d, m*n*sizeof(double)));
    
        // Copy input arrays to device
        cudaErrCheck(cudaMemcpy(a_d, a_h, m*k*sizeof(double), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(b_d, b_h, k*n*sizeof(double), cudaMemcpyHostToDevice));
    
        cublasOperation_t transa_int = (transa == 'N' || transa == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
        cublasOperation_t transb_int = (transb == 'N' || transb == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
    
        // Perform GEMM
        cublasErrCheck(
                cublasGemmEx(
                        cublasHandle, transa_int, transb_int,
                        m, n, k,
                        &alpha,
                        a_d, CUDA_R_64F, lda,
                        b_d, CUDA_R_64F, ldb,
                        &beta,
                        c_d, CUDA_R_64F, ldc,
                        CUDA_R_64F,
                        CUBLAS_GEMM_DEFAULT
                )
        );
    
        // Copy results back from device to host
        cudaErrCheck(cudaMemcpy(c_h, c_d, m*n*sizeof(double), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // Free memory on device
        cudaErrCheck(cudaFree(a_d));
        cudaErrCheck(cudaFree(b_d));
        cudaErrCheck(cudaFree(c_d));
    
        // =========================================================================
    
        // Set incoming C array pointer
        //c_p = (void *)c_h;
    }
}

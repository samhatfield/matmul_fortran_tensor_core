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

// Converts from double-precision to half-precision (CUDA kernel)
__global__ void double2half(half *out, const double *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half((float)(in[idx]));
    }
}

// Performs matrix-matrix multiplication using Tensor Core.
extern "C" {
    void tcgemm_c(int transa, int transb, int m, int n, int k, float alpha, void *a_p, int lda, void *b_p,
                int ldb, float beta, void *c_p, int ldc) {

        // Set up host-side arrays
        double *a_h, *b_h;
        float *c_h;
        a_h = (double *)a_p;
        b_h = (double *)b_p;
        c_h = (float *)c_p;

        // =========================================================================
        // Compute GEMM using Tensor Core
        // =========================================================================

        // Set up GPU and cuBLAS
        cublasHandle_t cublasHandle;
        cudaSetDevice(0);
        cudaDeviceReset();
        cublasErrCheck(cublasCreate(&cublasHandle));

        // Set up device-side arrays
        double *a_d, *b_d;
        half *a_d_16, *b_d_16;
        float *c_d_32;

        // Allocate memory on device for all arrays
        // TODO: should the dimensions used below (m*k etc.) take into account transa, lda etc.?
        cudaErrCheck(cudaMalloc((void **)&a_d, m*k*sizeof(double)));
        cudaErrCheck(cudaMalloc((void **)&b_d, k*n*sizeof(double)));
        cudaErrCheck(cudaMalloc((void**)&a_d_16, m*k*sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&b_d_16, k*n*sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&c_d_32, m*n*sizeof(float)));

        // Copy input arrays to device
        cudaErrCheck(cudaMemcpy(a_d, a_h, m*k*sizeof(double), cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(b_d, b_h, k*n*sizeof(double), cudaMemcpyHostToDevice));

        // Convert arrays to half-precision
        double2half<<<(int)((m*k)/256) + 1, 256 >>>(a_d_16, a_d, m*k);
        double2half<<<(int)((k*n)/256) + 1, 256 >>>(b_d_16, b_d, k*n);

        cudaDeviceSynchronize();

        // Perform GEMM with Tensor Core
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
        cublasErrCheck(
                cublasGemmEx(
                        cublasHandle, (cublasOperation_t)transa, (cublasOperation_t)transb,
                        m, n, k,
                        &alpha,
                        a_d_16, CUDA_R_16F, lda,
                        b_d_16, CUDA_R_16F, ldb,
                        &beta,
                        c_d_32, CUDA_R_32F, ldc,
                        CUDA_R_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP
                )
        );

        // Copy results back from device to host
        cudaErrCheck(cudaMemcpy(c_h, c_d_32, m*n*sizeof(float), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();

        // Free memory on device
        cudaErrCheck(cudaFree(a_d));
        cudaErrCheck(cudaFree(b_d));
        cudaErrCheck(cudaFree(a_d_16));
        cudaErrCheck(cudaFree(b_d_16));
        cudaErrCheck(cudaFree(c_d_32));
    }
}

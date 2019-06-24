# Default compilers
FC = gfortran
NVCC = gcc

matmul_test: matmul_test.o cublas_gemm_f.o cublas_gemm_c.o
	$(FC) matmul_test.o cublas_gemm_f.o cublas_gemm_c.o -o matmul_test #-L$(CUDA)/lib64 -lcudart -lcublas -o matmul_test

matmul_test.o: cublas_gemm_f.o

%.o: %.f90
	$(FC) -c $< -o $@

%.o: %.c
	$(NVCC) -c $< -std=c99

.PHONY: clean
clean:
	rm -f *.o matmul_test

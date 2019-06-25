# Default compilers
FC = gfortran
NVCC = nvcc

matmul_test: matmul_test.o cublas_gemm_f.o cublas_gemm_c.o
	$(FC) matmul_test.o cublas_gemm_f.o cublas_gemm_c.o -L$(CUDA)/lib64 -lcudart -lcublas -o matmul_test

matmul_test.o: cublas_gemm_f.o

%.o: %.f90
	$(FC) -c $< -o $@

%.o: %.cu
	$(NVCC) -c $<

.PHONY: clean
clean:
	rm -f *.o *.mod matmul_test

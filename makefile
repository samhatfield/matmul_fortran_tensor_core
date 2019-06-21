# Default compilers
FC = gfortran
NVCC = nvcc

matmul_test: matmul_test.o fortran.o
	$(FC) matmul_test.o fortran.o -L$(CUDA)/lib64 -lcudart -lcublas -o matmul_test

fortran.o: $(CUDA)/src/fortran.c
	$(NVCC) -DCUBLAS_GFORTRAN -c $(CUDA)/src/fortran.c -o fortran.o

%.o: %.f90
	$(FC) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o matmul_test

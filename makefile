# Default compiler
FC = gfortran

matmul: matmul.o
    $(FC) -o matmul matmul.o -I$(BLAS)/include -L$(BLAS)/lib

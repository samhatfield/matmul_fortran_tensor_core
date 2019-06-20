# Default compiler
FC = pgfortran

matmul: matmul.o
	$(FC) -o matmul matmul.o -lopenblas

%.o: %.f90
	$(FC) -c $< -o $@

.PHONY: clean
clean:
	rm -f *.o matmul

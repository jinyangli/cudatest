all: mem mpitest

mem.o: mem.cu
	nvcc -c -g mem.cu 

main.o: main.cc
	mpic++ -c -g main.cc

mem : mem.o main.o
	mpic++ main.o mem.o -L${CUDA_LIB} -lcudart -o mem

mpitest: mpitest.cc
	nvcc -I${MPI_HOME}/include mpitest.cc -L${MPI_HOME}/lib -lmpi -o mpitest

clean:
	rm -f mem mpitest *.o

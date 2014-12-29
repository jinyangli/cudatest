all: mem mpitest

mem.o: mem.cu
	nvcc -c -g mem.cu 

main.o: main.cc
	mpic++ -c -g main.cc

mem : mem.o main.o
	mpic++ main.o mem.o -L${CUDA_LIB} -lcudart -o mem

mpitest: mpitest.cc
	mpic++ mpitest.cc -o mpitest

clean:
	rm -f mem mpitest *.o

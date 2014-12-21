all: mem 

mem.o: mem.cu
	nvcc -c mem.cu 

main.o: main.cc
	mpicc -c main.cc

mem : mem.o main.o
	mpicc main.o mem.o -L/share/apps/cuda/6.5.12/lib64 -lcudart

mpitest: mpitest.cc
	nvcc mpitest.cc -lmpi -o mpitest

clean:
	rm -f mem mpitest

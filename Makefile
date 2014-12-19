all: mem mpitest

mem: mem.cu
	nvcc mem.cu -I${HOME}/pkg/mvapich2/include -L${HOME}/pkg/mvapich2/lib -lmpi -o mem 
mpitest: mpitest.cc
	nvcc mpitest.cc -I${HOME}/pkg/mvapich2/include -L${HOME}/pkg/mvapich2/lib -lmpi -o mpitest

clean:
	rm -f mem mpitest

all: mem

mem: mem.cu
	nvcc mem.cu -o mem

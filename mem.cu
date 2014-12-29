#include "util/cuPrintf.cu"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <assert.h>


#include "mem.h"
#include "timer.h"

extern int debug;


__global__  void kernel(float *array, int n, int stride)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; 

	int start = index * stride;
	int end = (index+1) * stride;
	if (end > n)
		end = n;

	for (int i = start; i < end; i++) {
		array[i] = sqrtf(array[i]);
	}

	cuPrintf("n %d stride %d my_id %d start %d end %d array[0]=%f\n", n, stride, index, start, end, array[0]);
}

void launch_kernel(int n_tblk, int nt_tblk, float *device, int n)
{	

	if (debug) cudaPrintfInit(); // initialize cuPrintf

	{
		Timer t("Kernel finished ");

		kernel<<<n_tblk,nt_tblk>>>(device, n, n/(n_tblk*nt_tblk));
		cudaDeviceSynchronize();
	}

	if (debug) {
		// display the device's greeting
		cudaPrintfDisplay();
			
		// clean up after cuPrintf
		cudaPrintfEnd();
	}

}

void alloc_mem(float **host_array, float **device_array, int n)
{
	cudaError_t err = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (err != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(err));
		exit(-1);
	}

	cudaMallocHost(host_array, n*sizeof(float));
	memset(*host_array, 0, n*sizeof(float));

	// cudaMalloc device memory
	//cudaMalloc(device_array, n* sizeof(float));
	assert(cudaHostGetDevicePointer(device_array, *host_array, 0) == cudaSuccess);

	// zero out the device array with cudaMemset
	cudaMemset(*device_array, 0, n* sizeof(float));

}

void transfer_mem(float *device, float *host, int n, bool host2dev)
{
	struct timespec t0, t1;

	clock_gettime(CLOCK_REALTIME, &t0);
	if (host2dev) 
		cudaMemcpy(device, host, n* sizeof(float), cudaMemcpyHostToDevice);
	else
		cudaMemcpy(host, device, n* sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("%s Transfer took %ld usec\n", host2dev?"H->D":"D->H", TIME_DIFF(t0, t1));
}

void copy_mem(float *dst, float *src, int n)
{
	struct timespec t0, t1;

	clock_gettime(CLOCK_REALTIME, &t0);
	cudaMemcpy(dst, src, n* sizeof(float), cudaMemcpyDefault);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &t1);

	struct cudaPointerAttributes attr;
	assert(cudaPointerGetAttributes(&attr, dst)==cudaSuccess);
	printf("%s Transfer took %ld usec\n", (attr.memoryType == cudaMemoryTypeHost)?"H->D":"D->H", TIME_DIFF(t0, t1));
}

void free_mem(float *host, float *device)
{
	cudaFreeHost(host);
	cudaFree(device);
}


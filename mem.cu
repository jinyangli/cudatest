#include "util/cuPrintf.cu"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>


#include "mem.h"
#include "timediff.h"

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
	struct timespec t0, t1;

	if (debug) cudaPrintfInit(); // initialize cuPrintf

	clock_gettime(CLOCK_REALTIME, &t0);
	kernel<<<n_tblk,nt_tblk>>>(device, n, n/(n_tblk*nt_tblk));
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("Kernel finished in %d usec\n", TIME_DIFF(t0, t1));

	if (debug) {
		// display the device's greeting
		cudaPrintfDisplay();
			
		// clean up after cuPrintf
		cudaPrintfEnd();
	}

}

void alloc_device_mem(float **device_array, int n)
{
	// cudaMalloc device memory
	cudaMalloc(device_array, n* sizeof(float));

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
	printf("%s Transfer took %d usec\n", host2dev?"H->D":"D->H", TIME_DIFF(t0, t1));
}

void free_device_mem(float *device)
{
	cudaFree(device);
}


#include "util/cuPrintf.cu"

#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

__global__  void kernel(float *array, int n, int stride)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; 

	int start = index * stride;
	int end = (index+1) * stride;
	if (end > n)
		end = n;

	cuPrintf("n %d stride %d my_id %d start %d end %d\n", n, stride, index, start, end);

	for (int i = start; i < end; i++) {
		array[i] = sqrtf(array[i]);
	}
}

int
timediff(struct timespec *t0, struct timespec *t1)
{
	return (t1->tv_sec - t0->tv_sec)*1000000 + (t1->tv_nsec - t0->tv_nsec)/1000;
}

int main(int argc, char **argv)
{
	bool debug = 0;
	int num_elements = 16;
	int n_tblk = 1;
	int nt_tblk = 1;
	char c;
	struct timespec t0, t1;

	while (( c = getopt(argc, argv, "d:n:b:t:")) != -1) {
		switch (c) 
		{
			case 'd':
				debug = atoi(optarg);
				break;
			case 'n':
				if (num_elements > 250) {
					num_elements = 250;
					printf("Capped to 250M elements!!!!\n");
				}
				num_elements = atoi(optarg) * 1000000;
				break;
			case 'b':
				n_tblk= atoi(optarg);
				break;
			case 't':
				nt_tblk= atoi(optarg);
				break;
			default:
				printf("valid options: -n <size> -b <# thread blocks> -t <# threads per block> -d <debug>\n");
				exit(1);
			}
	}

	printf("Array size:%dM, ThreadBlocks:%d, ThreadsPerBlock: %d Total Threads %d\n", \
			num_elements/1000000, n_tblk, nt_tblk, n_tblk * nt_tblk);

	float *device_array = 0;
	float *host_array = 0;

	// malloc host memory
	host_array = (float *)malloc(num_elements * sizeof(float));
	for (int i = 0; i < num_elements; i++) {
		host_array[i] = i + 4.0;
	}

	// cudaMalloc device memory
	cudaMalloc((void**)&device_array, num_elements * sizeof(float));

	// zero out the device array with cudaMemset
	cudaMemset(device_array, 0, num_elements * sizeof(float));

	// copy the contents of the device array to the host
	clock_gettime(CLOCK_REALTIME, &t0);
	cudaMemcpy(device_array, host_array, num_elements * sizeof(float), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("H->D Transfer %d usec\n", timediff(&t0, &t1));

	if (debug) {
		cudaPrintfInit(); // initialize cuPrintf
	}


	clock_gettime(CLOCK_REALTIME, &t0);
	kernel<<<n_tblk,nt_tblk>>>(device_array, num_elements, num_elements/(n_tblk*nt_tblk));
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &t1);

	printf("Kernel finished in %d usec\n", timediff(&t0, &t1));

	if (debug) {
		// display the device's greeting
		cudaPrintfDisplay();
			
		// clean up after cuPrintf
		cudaPrintfEnd();
	}

	clock_gettime(CLOCK_REALTIME, &t0);
	cudaMemcpy(host_array, device_array, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_REALTIME, &t1);
	printf("D->H Transfer %d usec\n", timediff(&t0, &t1));

	if (debug) {
		for(int i = 0; i < num_elements; ++i)
			printf("%f ", host_array[i]);
		printf("\n");
	}

	// use free to deallocate the host array
	free(host_array);

	// use cudaFree to deallocate the device array
	cudaFree(device_array);

	return 0;
}

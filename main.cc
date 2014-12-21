#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>

#include "mem.h"

int debug = 1;

int main(int argc, char **argv)
{
	bool debug = 1;
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

	int myrank;
	MPI_Status status;
	int tag = 999;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	float *device_array = 0;
	float *host_array = 0;

	// malloc host memory
	host_array = (float *)malloc(num_elements * sizeof(float));
	for (int i = 0; i < num_elements; i++) {
		host_array[i] = i + 4.0;
	}

	alloc_device_mem(&device_array, num_elements);
	
	if (myrank == 0 ) {
		// copy the contents of the device array to the host
		transfer_mem(device_array, host_array, num_elements, true);
		
		launch_kernel(n_tblk, nt_tblk, device_array, num_elements);

		transfer_mem(device_array, host_array, num_elements, false);
		
		MPI_Send(device_array, num_elements, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
		printf("MPI sent\n");
	}else {
		MPI_Recv(device_array, num_elements, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &status);
		int count;
		MPI_Get_count(&status, MPI_FLOAT, &count); 
		printf("MPI received %d floats %f\n", count, host_array[0]);
	}


	if (debug) {
		for(int i = 0; i < (num_elements<10?num_elements:10); ++i)
			printf("%f ", host_array[i]);
		printf("\n");
	}

	// use free to deallocate the host array
	free(host_array);


	free_device_mem(device_array);

	MPI_Finalize();

	return 0;
}

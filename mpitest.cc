#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <assert.h>

#include "timer.h"

int main(int argc, char *argv[])
{
	int myrank, tag=99;
	MPI_Status status;

	char hostname[256];
	gethostname(hostname, sizeof(hostname));

	/* Initialize the MPI library */
	MPI_Init(&argc, &argv);
	/* Determine unique id of the calling process of all processes participating
	   in this MPI program. This id is usually called MPI rank. */
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int msize = 1024, rsize = 1;
	int isends = 1;
	char c;
	while (( c = getopt(argc, argv, "m:r:i:")) != -1) {
		switch (c) 
		{
			case 'm':
				msize = atoi(optarg);
				msize *= 1024;
				break;
			case 'r':
				rsize = atoi(optarg);
				rsize *= 1024;
				break;
			case 'i':
				isends = atoi(optarg);
				break;
			default:
				printf("%s -s <message size in KB> -r <response size in KB>\n", argv[0]);
				exit(1);
		}
	}

	if (msize < isends)
		isends = msize;

	char *message = (char *)malloc(msize);
	assert(message);
	char *response = (char *)malloc(rsize);
	assert(response);

	MPI_Request *request = (MPI_Request *)malloc(isends*sizeof(MPI_Request));
	assert(request);
	memset(request, 0, isends*sizeof(MPI_Request));

	MPI_Status *reqstat = (MPI_Status *)malloc(isends*sizeof(MPI_Status));
	assert(reqstat);
	memset(reqstat, 0, isends*sizeof(MPI_Status));
	
	if (myrank == 0) {
		{
			Timer t("Ping pong ");
			for (int i = 0; i < isends; i++) {
				MPI_Isend(message+i*(msize/isends), msize/isends, MPI_CHAR, 1, tag, MPI_COMM_WORLD, request + i);
			}
			MPI_Waitall(isends, request, reqstat);
			MPI_Recv(response, rsize, MPI_CHAR, 1, tag, MPI_COMM_WORLD,
									                        &status);
			printf("Measured bandwidth %.2f GB/sec\n", ((msize+rsize))/((double)t.getLapsed()*1000.0));
		}
		int count;
		MPI_Get_count(&status, MPI_CHAR, &count); 
		printf("%s: received response %d\n", hostname, count);
	} else {
		for (int i = 0; i < isends; i++) {
			MPI_Irecv(message+i*(msize/isends), msize/isends, MPI_CHAR, 0, tag, MPI_COMM_WORLD, request + i);
		}
		MPI_Waitall(isends, request, reqstat);
		MPI_Send(response, rsize, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
	}

	free(message);
	free(response);
	/* Finalize the MPI library to free resources acquired by it. */
	MPI_Finalize();
	return 0;
}

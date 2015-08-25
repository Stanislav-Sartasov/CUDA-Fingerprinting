
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "stdlib.h"

#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess) {\
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(0);\
											}\
}

struct Lock
{
	int* mutex;
	Lock(void)
	{
		int state = 0;
		cudaMalloc((void**)&mutex, sizeof(int));
		cudaCheckError();
		cudaMemcpy(mutex, &state, sizeof(int), cudaMemcpyHostToDevice);
		cudaCheckError();
	}

	__device__ void lock()
	{
		while (atomicCAS(mutex, 0, 1) != 0);
	}

	__device__ void unlock()
	{
		atomicExch(mutex, 0);
	}

	~Lock(void)
	{
		cudaFree(mutex);
	}
};
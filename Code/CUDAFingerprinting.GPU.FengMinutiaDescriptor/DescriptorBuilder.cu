#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaStructs.cuh"
#include "MinutiaHelper.cuh"

#include <Math.h>


__global__ void buildDescriptors(Minutia **mins, int *minutiaNum, Descriptor **desc, int dbSize)//desc must be shared!
{/*
	int i, j, dev_j, num;
	float length;
	__shared__ int k;
	if (threadIdx.x == 0)
	{
		k = 0;
	}
	__syncthreads();
	num = blockIdx.x;
	i = blockIdx.y;
	j = threadIdx.x;

	if (blockIdx.x < dbSize && i < minutiaNum[num] && j < minutiaNum[num])
	{
		length = sqrLength(mins[num][i], mins[num][j]);
		if (i != j && length <= DESCRIPTOR_RADIUS*DESCRIPTOR_RADIUS)
		{
			dev_j = atomicAdd(&k, 1);
			desc[num][i].center = mins[num][i];
			desc[num][i].minutias[dev_j] = mins[num][j];
		}
	}*/
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaStructs.cuh"
#include "MinutiaHelper.cuh"

#include <Math.h>


__global__ void buildDescriptors(Minutia **mins, int *minutiaNum, Descriptor **desc)//desc must be shared!
{/*
	int i, j, dev_j, num;
	float length;
	__shared__ int k = 0;
	num = blockIdx.x;
	i = blockIdx.y;
	j = threadIdx.x;

	if (blockIdx.x >= MAX_DESC_SIZE*MAX_DESC_SIZE ||
		i >= minutiaNum[num] || j >= minutiaNum[num])
	{
		return;
	}

	length = sqrLength(mins[num][i], mins[num][j]);
	if (i != j && length <= DESCRIPTOR_RADIUS*DESCRIPTOR_RADIUS)
	{
		dev_j = atomicAdd(&k, 1);
		desc[num][i].center = mins[num][i];
		desc[num][i].minutias[dev_j] = mins[num][j];
	}
}*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaStructs.cuh"
#include "MinutiaHelper.cuh"

#include <Math.h>
#include <stdio.h>


__global__ void buildDescriptors(Minutia *mins, int pitch, int *minutiaNum, Descriptor *desc, int dbSize)
{
	int i, j, temp_j, num;
	float length;
	num = blockIdx.x;
	i = blockIdx.y;
	j = threadIdx.x;
	if (j == 0)
	{
		desc[num*pitch + i].length = 0;
	}
	__syncthreads();

	if (blockIdx.x < dbSize && i < minutiaNum[num] && j < minutiaNum[num])
	{
		length = sqrLength(mins[num*pitch + i], mins[num*pitch + j]);
		if (i != j && length <= DESCRIPTOR_RADIUS*DESCRIPTOR_RADIUS)
		{
			temp_j = atomicAdd(&(desc[num*pitch + i].length), 1);
			desc[num*pitch + i].center = mins[num*pitch + i];
			normalizeAngle(&(desc[num*pitch + i].center.angle));
			desc[num*pitch + i].minutias[temp_j] = mins[num*pitch + j];
			normalizeAngle(&(desc[num*pitch + i].minutias[temp_j].angle));
		}
	}
	__syncthreads();
}

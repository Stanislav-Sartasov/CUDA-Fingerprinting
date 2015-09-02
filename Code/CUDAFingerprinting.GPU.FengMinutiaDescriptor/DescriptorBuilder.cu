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

	if (num < dbSize && i < minutiaNum[num] && j < minutiaNum[num])
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

void buildFingerDescriptors(Minutia *mins, int *minutiaNum, Descriptor *desc)
{
	float length;
	int temp_j;

	for (int i = 0; i < MAX_DESC_SIZE; i++)
	{
		desc[i].length = 0;
	}

	for (int i = 0; i < *minutiaNum; i++)
	{
		for (int j = 0; j < *minutiaNum; j++)
		{
			length = sqrLength(mins[i], mins[j]);
			if (i != j && length <= DESCRIPTOR_RADIUS*DESCRIPTOR_RADIUS)
			{
				temp_j = desc[i].length;
				desc[i].length++;
				desc[i].center = mins[i];
				normalizeAngle(&(desc[i].center.angle));
				desc[i].minutias[temp_j] = mins[j];
				normalizeAngle(&(desc[i].minutias[temp_j].angle));
			}
		}
	}
}

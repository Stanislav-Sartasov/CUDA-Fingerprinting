#include "cuda_runtime.h"
#include "MinutiaHelper.cuh"
#include <stdio.h>

//#include "device_launch_parameters.h"

__device__ void transformate(Minutia* src, Minutia center, Minutia* dst, int j)
{
	float angle = center.angle - src[j].angle;
	float cosAngle = cos(angle);
	float sinAngle = sin(angle);
	float dx = src[j].x - center.x;
	float dy = src[j].y - center.y;

	int x = (int)round(dx * cosAngle + dy * sinAngle) + center.x;
	int y = (int)round(-dx * sinAngle + dy * cosAngle) + center.y;

	dst[j].angle = src[j].angle + angle;
	normalizeAngle(&(dst[j].angle));
	dst[j].x = x;
	dst[j].y = y;
}

__device__ void matchingPoints(Descriptor desc1, Descriptor desc2, int* m, int* M, int i, int j, int width, int height)
{
	float eps = 0.1;

	*m = 0;
	*M = 0;

	if ((sqrLength(desc1.minutias[i], desc2.minutias[j]) < COMPARE_RADIUS*COMPARE_RADIUS)
		&& ((desc1.minutias[i].angle - desc2.minutias[j].angle) < eps))
	{
		*m = 1;
		*M = 1;
	}
	else
	{
		if ((sqrLength(desc1.minutias[i], desc2.center) < FENG_CONSTANT * DESCRIPTOR_RADIUS * DESCRIPTOR_RADIUS) &&
			(desc1.minutias[i].x >= 0 && desc1.minutias[i].x < width
			&& desc1.minutias[i].y >= 0 && desc1.minutias[i].y < height))
		{
			*M = 1;
		}
	}
}

__global__ void compareDescriptors(Descriptor* input, Descriptor* current, Descriptor* temp0, Descriptor* temp1, 
	float* s, int height, int width, int pitch) //block 16*16*2
{ 
	__shared__ int cache_m[DESC_BLOCK_SIZE][DESC_BLOCK_SIZE][2];
	__shared__ int cache_M[DESC_BLOCK_SIZE][DESC_BLOCK_SIZE][2];

	int row = defaultRow();
	int column = defaultColumn();
	int x = defaultDescriptorRow();
	int y = defaultDescriptorColumn();
	int k = defaultFinger();
	
	int cacheIdxX = threadIdx.y;
	int cacheIdxY = threadIdx.x;
	int cacheIdxZ = threadIdx.z;

	float eps = 0.1f;
	
	
	if (row == 0 && column == 0 && k == 0 && cacheIdxZ == 0)
	{
		for (int i = 0; i < 10; i++)
		{
			printf("%d\n", input[i].length);
			for (int j = 0; j < input[i].length; j++)
			{
				//printf("%d %d %f\n", input[i].minutias[j].x, input[i].minutias[j].y, input[i].minutias[j].angle);
			}
		}
		
	}
	/*
	if ((cacheIdxX == 0) && (cacheIdxZ == 0))
	{
		transformate(input[x].minutias, current[k*pitch + y].center, temp0[k*pitch + y].minutias, column);
	}
	else if ((cacheIdxX == 0) && (cacheIdxZ == 1))
	{
		transformate(current[k*pitch + y].minutias, input[x].center, temp1[k*pitch + y].minutias, column);
	}
	__syncthreads();
	

	if (cacheIdxZ == 0)
	{
		matchingPoints(temp0[k*pitch + y], current[k*pitch + y], &cache_m[cacheIdxX][cacheIdxY][0],
			&cache_M[cacheIdxX][cacheIdxY][0], row, column, width, height);
	}
	else
	{
		matchingPoints(temp1[k*pitch + y], input[x], &cache_m[cacheIdxX][cacheIdxY][1],
			&cache_M[cacheIdxX][cacheIdxY][1], row, column, width, height);
	}

	__syncthreads();

	int i = DESC_BLOCK_SIZE / 2;
	while (i != 0)
	{
		if (cacheIdxX < i)
		{
			cache_m[cacheIdxX][cacheIdxY][cacheIdxZ] += cache_m[cacheIdxX + i][cacheIdxY][cacheIdxZ];
		}
		else 
		{
			cache_M[cacheIdxX - i][cacheIdxY][cacheIdxZ] += cache_m[cacheIdxX][cacheIdxY][cacheIdxZ];
		}

		__syncthreads();
		i /= 2;
	}

	i = DESC_BLOCK_SIZE / 2;
	while (i != 0)
	{
		if (cacheIdxX == 0)
		{
			if (cacheIdxY < i)
			{
				cache_m[cacheIdxX][cacheIdxY][cacheIdxZ] += cache_m[cacheIdxX][cacheIdxY + i][cacheIdxZ];
			}
			else
			{
				cache_M[cacheIdxX][cacheIdxY][cacheIdxZ] += cache_m[cacheIdxX][cacheIdxY - i][cacheIdxZ];
			}
		}

		__syncthreads();
		i /= 2;
	}

	if ((cacheIdxX == 0) && (cacheIdxY == 0) && (cacheIdxZ == 0))
	{
		s[k*MAX_DESC_SIZE + blockDim.y*blockIdx.y + blockIdx.x] = (cache_m[0][0][0] + 1.0f)*(cache_m[0][0][1] + 1.0f) / (cache_M[0][0][0] + 1.0f) / (cache_M[0][0][1] + 1.0f);
	}*/
}


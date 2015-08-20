#include "cuda_runtime.h"
#include "MinutiaHelper.cuh"

__global__ void makeNormalSize(float*** s, float*** sn) ///block 8*8
{
	__shared__ float sum[DESC_PER_BLOCK][DESC_PER_BLOCK];
	
	int k = defaultFinger();

	sum[threadIdx.y][threadIdx.x] = s[k][threadIdx.y][threadIdx.x];

	int i = DESC_PER_BLOCK / 2;
	while (i != 0)
	{
		if (threadIdx.y < i)
		{
			sum[threadIdx.y][threadIdx.x] += sum[threadIdx.y + i][threadIdx.x];
		}

		__syncthreads();
		i /= 2;
	}

	i = DESC_PER_BLOCK / 2;
	while (i != 0)
	{
		if (threadIdx.y == 0)
		{
			if (threadIdx.x < i)
			{
				sum[threadIdx.y][threadIdx.x] += sum[threadIdx.y][threadIdx.x + i];
			}
		}

		__syncthreads();
		i /= 2;
	}

	if ((threadIdx.y == 0) && (threadIdx.x == 0))
	{
		sn[k][blockIdx.y][blockIdx.x] = sum[0][0];
	}
}

__global__ void normalize(float*** s, Minutia* input, Minutia** current)
{

}
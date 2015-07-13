#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "constsmacros.h"
#include <stdlib.h>
#include "CUDAArray.cuh"
#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "CUDAArray.cuh"

extern "C"
{
	__declspec(dllexport) void MakeGaborFilters(float* filter, int size, int angleNum, float frequency);
}

__global__ void cudaCreateGaborFilter(CUDAArray<float> filters, int size, float frequency, float bAngle)
{
	float aCos = cos(M_PI / 2 + bAngle * (blockIdx.x));
	float aSin = sin(M_PI / 2 + bAngle * (blockIdx.x));

	int center = size / 2;
	int upperCenter = (size & 1) == 1 ? center - 1 : center;

	for (int i = -upperCenter; i < center; i++)
	{
		for (int j = -upperCenter; j < center; j++)
		{
			filters.SetAt(blockIdx.x * blockDim.x + center - i , center - j, exp(-0.5 * ((i * aSin + j * aCos) * (i * aSin + j * aCos) / 16 + (-i *aCos + j * aSin) * (-i *aCos + j * aSin) / 16)) * cos(2 * M_PI * (i * aSin + j * aCos) * frequency));
		}
	}
}

void MakeGaborFilters(float* filter, int size, int angleNum, float frequency)
{
	CUDAArray<float> filters = CUDAArray<float>(size, size * angleNum);
	
	float bAngle = (float) M_PI / angleNum;

	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(angleNum, defaultThreadCount));

	cudaCreateGaborFilter << <gridSize, blockSize >> > (filters, size, frequency, bAngle);

	filters.GetData(filter);
}
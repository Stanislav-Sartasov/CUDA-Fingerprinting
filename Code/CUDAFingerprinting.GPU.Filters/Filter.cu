#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "constsmacros.h"
#include <stdlib.h>
#include "CUDAArray.cuh"
#include <float.h>
#include "math_constants.h"
#include "math_functions.h"
#include "CUDAArray.cuh"
#include <math.h>
#include "Filter.cuh"

__global__ void cudaCreateGaborFilter(CUDAArray<float> filters, int size, float frequency, float bAngle)
{
	float aCos = cos(/*CUDART_PI_F / 2 +*/ bAngle * (blockIdx.x));
	float aSin = sin(/*CUDART_PI_F / 2 +*/ bAngle * (blockIdx.x));

	int center = size / 2;

	int dX = threadIdx.x - center;
	int dY = threadIdx.y - center;

	float xDash = dX * aSin + dY * aCos;
	float yDash = -dX *aCos + dY * aSin;
	float cellExp = exp(-0.5 * (xDash * xDash / 16 + yDash * yDash / 16));
	float cellCos = cos(2 * CUDART_PI_F * xDash * frequency);

	filters.SetAt(threadIdx.x, blockDim.x * blockIdx.x + threadIdx.y, cellExp * cellCos);
}

CUDAArray<float> MakeGabor16Filters(int angleNum, float frequency)
{
	CUDAArray<float> filters = CUDAArray<float>(16 * angleNum, 16);
	
	float bAngle = (float) CUDART_PI_F / angleNum;

	cudaCreateGaborFilter << < dim3(angleNum), dim3(16, 16) >> > (filters, 16, frequency, bAngle);

	return filters;
}

CUDAArray<float> MakeGabor32Filters(int angleNum, float frequency)
{
	CUDAArray<float> filters = CUDAArray<float>(32 * angleNum, 32);

	float bAngle = (float)CUDART_PI_F / angleNum;

	cudaCreateGaborFilter << < dim3(angleNum), dim3(32, 32) >> > (filters, 32, frequency, bAngle);

	return filters;
}

//int main()
//{
//	float* b = (float*)malloc(16*16*8*sizeof(float));
//
//	MakeGabor16Filters(b, 8, (float) 1 / 9);
//
//	return 0;
//}
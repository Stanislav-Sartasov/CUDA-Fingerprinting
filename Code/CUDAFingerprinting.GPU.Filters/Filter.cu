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
#include "ImageLoading.cuh"

__global__ void cudaCreateGaborFilter(CUDAArray<float> filters, int size, float* frequencyArr, float bAngle)
{
	float aCos = cos(bAngle * blockIdx.x);
	float aSin = sin(bAngle * blockIdx.x);

	int center = size / 2;

	int dX = threadIdx.x - center;
	int dY = threadIdx.y - center;

	float xDash = dX * aSin + dY * aCos;
	float yDash = -dX *aCos + dY * aSin;
	float cellExp = exp(-0.5 * (xDash * xDash / 16 + yDash * yDash / 16));
	float cellCos = cos(2 * CUDART_PI_F * xDash * frequencyArr[blockIdx.y]);
	filters.SetAt(threadIdx.x + blockIdx.y * size, blockDim.x * blockIdx.x + threadIdx.y, cellExp * cellCos);
}

//CUDAArray<float> MakeGabor16Filters(int angleNum, float frequency)
//{
//	CUDAArray<float> filters = CUDAArray<float>(16 * angleNum, 16);
//	
//	float bAngle = (float) CUDART_PI_F / angleNum;
//
//	cudaCreateGaborFilter << < dim3(angleNum), dim3(16, 16) >> > (filters, 16, frequency, bAngle);
//
//	return filters;
//}
//
//CUDAArray<float> MakeGabor32Filters(int angleNum, float frequency)
//{
//	CUDAArray<float> filters = CUDAArray<float>(32 * angleNum, 32);
//
//	float bAngle = (float)CUDART_PI_F / angleNum;
//
//	cudaCreateGaborFilter << < dim3(angleNum), dim3(32, 32) >> > (filters, 32, frequency, bAngle);
//
//	return filters;
//}

CUDAArray<float> MakeGaborFilters(int size, int angleNum, float* frequencyArr, int frNum)  //For creating filters of all sizes. This function isn't used.
{
	CUDAArray<float> filters = CUDAArray<float>(size * angleNum, size * frNum);

	float bAngle = (float)CUDART_PI_F / angleNum;
	float* dev_frArr;
	cudaMalloc((void**)&dev_frArr, frNum * sizeof(float));
	cudaMemcpy(dev_frArr, frequencyArr, frNum * sizeof(float), cudaMemcpyHostToDevice);
	cudaCreateGaborFilter << < dim3(angleNum, frNum), dim3(size, size) >> > (filters, size, dev_frArr, bAngle);
	return filters;
}

float Gaussian2D(float x, float y, float sigma)
{
	float commonDenom = 2.0 * sigma * sigma;
	float denominator = CUDART_PI_F * commonDenom;
	float result = exp(-(x * x + y * y) / commonDenom) / denominator;
	return result;
}

float* MakeGaussianFilter(int size, float sigma)
{
	float* filter = (float*)malloc(size * size * sizeof(float));

	int center = size / 2;
	int upperCenter = (size & 1) == 0 ? center - 1 : center;

	for (int i = -upperCenter; i <= upperCenter; i++)
	{
		for (int j = -upperCenter; j <= upperCenter; j++)
		{
			filter[(center - i) * size + (center - j)] = Gaussian2D(i, j, sigma);
		}
	}

	return filter;
}

//int main()
//{
//	float* b = (float*)malloc(16*16*8*sizeof(float));
//
//	MakeGabor16Filters(b, 8, (float) 1 / 9);
//
//	return 0;
//}
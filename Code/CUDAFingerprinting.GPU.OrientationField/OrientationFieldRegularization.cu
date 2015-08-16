#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
#include "OrientationFieldRegularization.cuh"
using namespace std;
static void HandleError(cudaError_t err, const char *file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
			file, line);
		exit(EXIT_FAILURE);
	}
}
__device__ float Gaussian2D(int x, int y, float sigma)
{
	float commonDenom = (float)2.0 * sigma * sigma;
	float denominator = (float)M_PI * commonDenom;
	float result = (float)exp(-(x * x + y * y) / commonDenom) / denominator;
	return result;
}
int CeilMod(int x, int y){
	return (x > 512 ? (x + y - 1) / y : x);
}
__global__ void VectorField(float *Fx, float *Fy, float *O, int width)
{
	if (blockIdx.y*blockDim.x + threadIdx.x < width)
	{
		int tid = blockIdx.y*blockDim.x + threadIdx.x + blockIdx.x*width;
		Fx[tid] = cos(2 * O[tid]);
		Fy[tid] = sin(2 * O[tid]);
	}
}
__global__ void Filter(float *Fx, float *Fy, float *Fx1, float *Fy1, int width)
{
	const int wf = 25;
	if (blockIdx.y*blockDim.x + threadIdx.x < width)
	{
		int tid = blockDim.x*blockIdx.y + threadIdx.x + blockIdx.x*width;
		int halfWf = ((wf % 2) == 0 ? wf / 2 - 1 : wf / 2);
		float W[wf*wf];
		float sigma = (float)4.0;
		for (int i = 0; i <wf; i++)
			for (int j = 0; j <wf; j++)
				W[i*wf + j] = Gaussian2D(halfWf - i, halfWf - j, sigma);
		Fx1[tid] = 0;
		Fy1[tid] = 0;
		for (int u = -wf / 2; u <= halfWf; u++)
			for (int v = -wf / 2; v <= halfWf; v++)
			{
			int i1 = blockIdx.x - u;
			int j1 = blockIdx.y*blockDim.x + threadIdx.x - v;
			int tidFil = (u + wf / 2)*wf + v + wf / 2;
			if ((i1 >= 0) & (i1 < gridDim.x) & (j1 >= 0) & (j1 < width))
			{
				Fx1[tid] += W[tidFil] * Fx[i1 * width + j1];
				Fy1[tid] += W[tidFil] * Fy[i1 * width + j1];
			}
			}
	}
}
__global__ void LocalOrientation(float *O, float *Fx1, float *Fy1, int width)
{
	if (blockIdx.y*blockDim.x + threadIdx.x < width)
	{
		int tid = blockDim.x*blockIdx.y + threadIdx.x + blockIdx.x*width;
		int sign;
		if (Fx1[tid] <= 0 && Fy1[tid] >= 0) sign = 1;
		else if (Fx1[tid] <= 0 && Fy1[tid] <= 0) sign = -1;
		else sign = 0;
		O[tid] = 0.5 *(atan(Fy1[tid] / Fx1[tid]) + sign*M_PI);
	}
}
void OrientationRegularizationPixels(float *Out, float* O, int height, int width)
{
	float *dev_Fy, *dev_Fx, *dev_O, *dev_Fy1, *dev_Fx1;
	int countThr = CeilMod(width, 16);
	HandleError(cudaMalloc((void**)&dev_Fx, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_Fy, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_O, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMemcpy(dev_O, O, (sizeof(float)*height*width), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	VectorField << < dim3(height, CeilMod(width, countThr)), countThr >> >(dev_Fx, dev_Fy, dev_O, width);
	HandleError(cudaMalloc((void**)&dev_Fx1, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_Fy1, (sizeof(float)*height*width)), __FILE__, __LINE__);
	Filter << < dim3(height, CeilMod(width, countThr)), countThr >> >(dev_Fx, dev_Fy, dev_Fx1, dev_Fy1, width);
	cudaFree(dev_Fx);
	cudaFree(dev_Fy);
	LocalOrientation << < dim3(height, CeilMod(width, countThr)), countThr >> >(dev_O, dev_Fx1, dev_Fy1, width);
	HandleError(cudaMemcpy(Out, dev_O, sizeof(float)*height*width, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	cudaFree(dev_Fy1);
	cudaFree(dev_Fx1);
	cudaFree(dev_O);
}
float* OrientationRegularizationPixels(float* O, int height, int width)
{
	float *dev_Fy, *dev_Fx, *dev_O, *dev_Fy1, *dev_Fx1;
	int countThr = CeilMod(width, 16);
	float *Out = (float*)malloc(height*width*sizeof(float));
	HandleError(cudaMalloc((void**)&dev_Fx, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_Fy, (sizeof(float)*height*width)), __FILE__, __LINE__);;
	HandleError(cudaMalloc((void**)&dev_O, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMemcpy(dev_O, O, (sizeof(float)*height*width), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	VectorField << < dim3(height, CeilMod(width, countThr)), countThr >> >(dev_Fx, dev_Fy, dev_O, width);
	HandleError(cudaMalloc((void**)&dev_Fx1, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_Fy1, (sizeof(float)*height*width)), __FILE__, __LINE__);
	Filter << < dim3(height, CeilMod(width, countThr)), countThr >> >(dev_Fx, dev_Fy, dev_Fx1, dev_Fy1, width);
	cudaFree(dev_Fx);
	cudaFree(dev_Fy);
	LocalOrientation << < dim3(height, CeilMod(width, countThr)), countThr >> >(dev_O, dev_Fx1, dev_Fy1, width);
	HandleError(cudaMemcpy(Out, dev_O, sizeof(float)*height*width, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	cudaFree(dev_Fy1);
	cudaFree(dev_Fx1);
	cudaFree(dev_O);
	return Out;
}
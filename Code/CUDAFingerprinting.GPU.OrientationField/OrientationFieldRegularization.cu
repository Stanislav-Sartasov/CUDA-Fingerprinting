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
int CeilMod(int x, int y){
	return (x > 1024 ? (x + y - 1) / y : x);
}
__global__ void NewFilter(float* filter, float sigma)
{
	float commonDenom = (float)2.0 * sigma * sigma;
	float denominator = (float)M_PI * commonDenom;
	int x = (blockDim.x >> 1) - blockIdx.x;
	int y = (blockDim.x >> 1) - threadIdx.x;
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	filter[tid] = (float)(exp((float)(-((float)(x*x + y*y)) / commonDenom)) / denominator);
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
__global__ void FilterAndLocalOrient(float* O, float *Fx, float *Fy, int width, int sizeFil, float* filter)
{
	if (blockIdx.y*blockDim.x + threadIdx.x < width)
	{
		int tid = blockDim.x*blockIdx.y + threadIdx.x + blockIdx.x*width;
		float resX = 0;
		float resY = 0;
		int i1 = blockIdx.x + (sizeFil >> 1);
		for (int u = 0; u < sizeFil; u++, i1--){
			int j1 = blockIdx.y*blockDim.x + threadIdx.x + (sizeFil >> 1);
			for (int v = 0; v < sizeFil; v++, j1--)
			{
				if ((i1 >= 0) && (i1 < gridDim.x) && (j1 >= 0) && (j1 < width))
				{
					int tidFil = u*sizeFil + v;
					resX += filter[tidFil] * Fx[i1 * width + j1];
					resY += filter[tidFil] * Fy[i1 * width + j1];
				}
			}
		}
		O[tid] = (float)0.5 *(atan(resY / resX));
		if (resX <= 0 && resY >= 0) O[tid] += M_PI_2;
		else if (resX <= 0 && resY <= 0) O[tid] -= M_PI_2;
	}
}
void OrientationRegularizationPixels(float *Out, float* O, int height, int width, int sizeFil)
{
	float *dev_Fy, *dev_Fx, *dev_O;
	int countThr = CeilMod(width, 4);
	float sigma = float(sizeFil - 1) / 6;
	if (sizeFil % 2 == 0)
		sizeFil -= 1;
	float *dev_filter;
	HandleError(cudaMalloc((void**)&dev_filter, (sizeof(float)*sizeFil*sizeFil)), __FILE__, __LINE__);
	NewFilter << <sizeFil, sizeFil >> >(dev_filter, sigma);
	HandleError(cudaMalloc((void**)&dev_Fx, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_Fy, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_O, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMemcpy(dev_O, O, (sizeof(float)*height*width), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	VectorField << < dim3(height, (width + countThr - 1) / countThr), countThr >> >(dev_Fx, dev_Fy, dev_O, width);
	FilterAndLocalOrient << < dim3(height, (width + countThr - 1) / countThr), countThr >> >(dev_O, dev_Fx, dev_Fy, width, sizeFil, dev_filter);
	cudaFree(dev_Fx);
	cudaFree(dev_Fy);
	HandleError(cudaMemcpy(Out, dev_O, sizeof(float)*height*width, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

	cudaFree(dev_O);
	cudaFree(dev_filter);
}
float* OrientationRegularizationPixels(float* O, int height, int width, int sizeFil)
{
	float *dev_Fy, *dev_Fx, *dev_O;
	int countThr = CeilMod(width, 4);
	float sigma = float(sizeFil - 1) / 6;
	if (sizeFil % 2 == 0)
		sizeFil -= 1;
	float *dev_filter;
	HandleError(cudaMalloc((void**)&dev_filter, (sizeof(float)*sizeFil*sizeFil)), __FILE__, __LINE__);
	NewFilter << <sizeFil, sizeFil >> >(dev_filter, sigma);
	HandleError(cudaMalloc((void**)&dev_Fx, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_Fy, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMalloc((void**)&dev_O, (sizeof(float)*height*width)), __FILE__, __LINE__);
	HandleError(cudaMemcpy(dev_O, O, (sizeof(float)*height*width), cudaMemcpyHostToDevice), __FILE__, __LINE__);
	VectorField << < dim3(height, (width + countThr - 1) / countThr), countThr >> >(dev_Fx, dev_Fy, dev_O, width);
	FilterAndLocalOrient << < dim3(height, (width + countThr - 1) / countThr), countThr >> >(dev_O, dev_Fx, dev_Fy, width, sizeFil, dev_filter);
	cudaFree(dev_Fx);
	cudaFree(dev_Fy);
	float *Out = (float*)malloc(height*width*sizeof(float));
	HandleError(cudaMemcpy(Out, dev_O, sizeof(float)*height*width, cudaMemcpyDeviceToHost), __FILE__, __LINE__);
	cudaFree(dev_O);
	cudaFree(dev_filter);
	return Out;
}
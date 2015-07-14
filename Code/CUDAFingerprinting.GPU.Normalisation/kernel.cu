#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "constsmacros.h"
#include <stdlib.h>
#include <math.h>
#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
#include <float.h>

extern "C"
{ 
	__declspec(dllexport) void Normalize(float* source, float* res, int imgWidth, int imgHeight, int bordMean, int bordVar);
}
__global__ void cudaCalcMeanRow(CUDAArray<float> image, float* meanArray)
{
	
	int column = defaultColumn();
	__shared__ int height;
	__shared__ int width;
	__shared__ int pixNum;
	height = image.Height;
	width = image.Width;
	pixNum = height * width;
	int tempIndex = threadIdx.x;
	
	__shared__ float temp[defaultThreadCount];
	float mean = 0;
	if (width > column)
	{
		for (int j = 0; j < height; j++)
		{
			mean += image.At(j, column) / pixNum;
		}
	}
	
	temp[tempIndex] = mean;
	__syncthreads();
	
	//This is reduction.It will work only if number of threads in the block is a power of 2.
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i) 
			temp[tempIndex] += temp[tempIndex + i];
		i /= 2;
	}
	if (tempIndex == 0) 
		meanArray[blockIdx.x] = temp[0];//we need to write it only one time. Why not to choose the first thread for this purpose?
		
}


float CalculateMean(CUDAArray<float> image)
{
	int height = image.Height;
	float *dev_mean, mean = 0;
	
	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(height, defaultThreadCount));
	float* meanArray = (float*)malloc(gridSize.x * sizeof(float));
	cudaMalloc((void**)&dev_mean, gridSize.x * sizeof(float));

	cudaCalcMeanRow <<<gridSize, blockSize >>> (image, dev_mean);
	cudaMemcpy(meanArray, dev_mean, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < gridSize.x; i++)
	{
		mean += meanArray[i];
	}
	return mean;
}

__global__ void cudaCalcVariationRow(CUDAArray<float> image, float mean, float* variationArray)
{

	int column = defaultColumn();
	__shared__ int height;
	__shared__ int width;
	__shared__ int pixNum;
    height = image.Height;
	width = image.Width;
	pixNum = height * width;

	int tempIndex = threadIdx.x;

	__shared__ float temp[defaultThreadCount];
	float variation = 0;
	if (width > column)
	{
		for (int j = 0; j < height; j++)
		{
			float diff = image.At(j, column) - mean;
			variation += diff * diff / pixNum;
		}
	}
	temp[tempIndex] = variation;
	__syncthreads();
	//This is reduction.It will work only if number of threads in the block is a power of 2.
	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
			temp[tempIndex] += temp[tempIndex + i];
		i /= 2;
	}
	if (tempIndex == 0)
		variationArray[blockIdx.x] = temp[0];//we need to write it only one time. Why not to choose the first thread for this purpose?
}

float CalculateVariation(CUDAArray<float> image, float mean)
{
	int height = image.Height;
	float *dev_variation, variation = 0;

	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(height, defaultThreadCount));
	float* variationArray = (float*)malloc(gridSize.x * sizeof(float));
	cudaMalloc((void**)&dev_variation, gridSize.x * sizeof(float));

	cudaCalcVariationRow <<<gridSize, blockSize >>> (image, mean, dev_variation);
	cudaMemcpy(variationArray, dev_variation, gridSize.x * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < gridSize.x; i++)
	{
		variation += variationArray[i];
	}
	return variation;
}
__global__ void cudaDoNormalizationRow(CUDAArray<float> image, float mean, float variation, int bordMean, int bordVar)
{
	int column = defaultColumn();
	__shared__ int width;
	__shared__ int height;
	width = image.Width;
	height = image.Height;
	int curPix;  
	if (width > column)
	{
		for (int j = 0; j < height; j++)
		{
			curPix = image.At(j, column);
			if (curPix > mean)
			{
				image.SetAt(j, column, bordMean + sqrt((bordVar * (curPix - mean) * (curPix - mean)) / variation));
			}
			else
			{
				image.SetAt(j, column, bordMean - sqrt((bordVar * (curPix - mean) * (curPix - mean)) / variation));
			}
		}
	}
	
}

CUDAArray<float> Normalize(CUDAArray<float> image, int bordMean, int bordVar)
{
	int height = image.Height;

	float mean = CalculateMean(image);
	float variation = CalculateVariation(image, mean);

	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(height, defaultThreadCount));
	cudaDoNormalizationRow <<<gridSize, blockSize >>> (image, mean, variation, bordMean, bordVar);
	return image;
}

void Normalize(float* source, float* res, int imgWidth, int imgHeight, int bordMean, int bordVar)
{
	CUDAArray<float> image = CUDAArray<float>(source, imgWidth, imgHeight);
	int height = image.Height;

	float mean = CalculateMean(image);
	float variation = CalculateVariation(image, mean);

	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(height, defaultThreadCount));
	cudaDoNormalizationRow <<<gridSize, blockSize >>> (image, mean, variation, bordMean, bordVar);
	image.GetData(res);
}

//void main()
//{
//	int width;
//	int height;
//	char* filename = "C:\\Users\\Alexander\\Documents\\CUDA-Fingerprinting\\Code\\CUDAFingerprinting.GPU.Normalisation\\002.bmp";  //Write your way to bmp file
//	int* img = loadBmp(filename, &width, &height);
//	float* source = (float*)malloc(height*width*sizeof(float));
//	for (int i = 0; i < height; i++)
//		for (int j = 0; j < width; j++)
//		{
//			source[i * width + j] = (float)img[i * width + j];
//		}
//	float* b = (float*)malloc(height * width * sizeof(float));
//	cudaEvent_t     start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	cudaEventRecord(start, 0);
//	Normalize(source, b, width, height, 200, 1000);
//	cudaEventRecord(stop, 0);
//	cudaEventSynchronize(stop);
//	float   elapsedTime;
//	cudaEventElapsedTime(&elapsedTime, start, stop);
//	printf("Time to generate:  %3.1f ms\n", elapsedTime);
//	cudaEventDestroy(start);
//	cudaEventDestroy(stop);
//	saveBmp("..\\res.bmp", b, width, height);
//
//	free(source);
//	free(img);
//	free(b);
//}

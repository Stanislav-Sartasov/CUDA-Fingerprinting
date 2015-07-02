#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>
#include "CUDAArray.cuh"
__global__ void cudaCalcMeanRow(CUDAArray<float> source, float* meanArray)
{
	
	int row = defaultRow();
	int height = source.Height;
	int width = source.Width;

	__shared__ float* temp;
	temp = (float*)malloc(sizeof(float));
	float mean = 0;
	if (source.Height > row)
	{
		for (int j = 0; j < source.Width; j++)
		{
			mean += source.At(row, j) / (height * width);
		}
	}
	temp[blockIdx.y] = mean;
	__syncthreads();//is it really necessary?
	meanArray = temp;
}

float CalculateMean(CUDAArray<float> image)
{
	int height = image.Height;
	int width  = image.Width;
	float *dev_mean, *meanArray, mean = 0;
	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize  = dim3(ceilMod(image.Height, defaultThreadCount));
	cudaCalcMeanRow <<<gridSize, blockSize >>> (image, dev_mean); 
	cudaMemcpy(meanArray, dev_mean, height * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
	{
		mean += meanArray[i];
	}
	return mean;
}

__global__ void cudaCalcVariationRow(CUDAArray<float> image, float mean, float* variationArray)
{

	int row = defaultRow();
	int height = image.Height;
	int width = image.Width;

	__shared__ float* temp;
	temp = (float*) malloc (sizeof(float));
	float variation = 0;
	if (image.Height > row)
	{
		for (int j = 0; j < image.Width; j++)
		{
			variation += pow((image.At(row, j) - mean), 2) / (height * width);
		}
	}
	temp[blockIdx.y] = variation;
	__syncthreads();//is it really necessary?
	variationArray = temp;
}

float CalculateVariation(CUDAArray<float> image, float mean)
{
	int height = image.Height;
	float *dev_variation, *variationArray, variation = 0;
	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(height, defaultThreadCount));
	cudaCalcVariationRow <<<gridSize, blockSize >>> (image, mean, dev_variation);
	cudaMemcpy(variationArray, dev_variation, height * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < height; i++)
	{
		variation += variationArray[i];
	}
	return variation;
}
/*
float CalculateVariation(CUDAArray<float> image, float mean)
{
	int height = image.Height;
	int width = image.Width;
	float variation = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			variation += pow((image.At(i, j) - mean), 2) / (height * width);
		}
	}
	return variation;
}

CUDAArray<float> DoNormalization(CUDAArray<float> image, int bordMean, int bordVar)
{
	float mean = CalculateMean(image);
	float variation = CalculateVariation(image, mean);

	for (int i = 0; i < image.Width; i++)
	{
		for (int j = 0; j < image.Height; j++)
		{
			if (image.At(i, j) > mean)
			{
				image.SetAt(i, j, bordMean + sqrt((bordVar * pow(image.At(i, j) - mean, 2)) / variation));
			}
			else
			{
				image.SetAt(i, j), bordMean - sqrt((bordVar * pow(image.At(i, j) - mean, 2)) / variation));
			}
		}
	}

	return image;
	*/
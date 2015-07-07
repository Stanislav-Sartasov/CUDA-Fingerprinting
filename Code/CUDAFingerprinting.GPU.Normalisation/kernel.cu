#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "constsmacros.h"
#include <stdlib.h>
#include <math.h>
#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
extern "C"
{ 
	__declspec(dllexport) void Normalize(float* source, float* res, int imgWidth, int imgHeight, int bordMean, int bordVar);
}
__global__ void cudaCalcMeanRow(CUDAArray<float> source, float* meanArray)
{
	
	int row = defaultColumn();
	int height = source.Height;
	int width = source.Width;
	int tempIndex = threadIdx.x;
	
	__shared__ float* temp;
	temp = (float*)malloc(blockDim.x * sizeof(float));
	float mean = 0;
	if (source.Height > row)
	{
		for (int j = 0; j < source.Width; j++)
		{
			mean += source.At(row, j) / (height * width);
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
		__syncthreads();
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

	int row = defaultColumn();
	int height = image.Height;
	int width = image.Width;
	int tempIndex = threadIdx.x;

	__shared__ float* temp;
	temp = (float*)malloc(blockDim.x * sizeof(float));
	float variation = 0;
	if (image.Height > row)
	{
		for (int j = 0; j < image.Width; j++)
		{
			variation += pow((image.At(row,j)- mean), 2) / (height * width);
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
		__syncthreads();
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
	int row = defaultColumn();

	if (image.Height > row)
	{
		for (int j = 0; j < image.Width; j++)
		{
			if (image.At(row, j) > mean)
			{
				image.SetAt(row, j, bordMean + sqrt((bordVar * pow(image.At(row, j) - mean, 2)) / variation));
			}
			else
			{
				image.SetAt(row, j, bordMean - sqrt((bordVar * pow(image.At(row, j) - mean, 2)) / variation));
			}
		}
	}
}

CUDAArray<float> DoNormalization(CUDAArray<float> image, int bordMean, int bordVar)
{
	int height = image.Height;

	float mean = CalculateMean(image);
	float variation = CalculateVariation(image, mean);

	dim3 blockSize = dim3(defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(height, defaultThreadCount));
	cudaDoNormalizationRow <<<gridSize, blockSize >>> (image, mean, variation, bordMean, bordVar);
	return image;
}
float* Make1D(float *arr, int width, int height)
{
	//var rows = arr.GetLength(0);
	//	var columns = arr.GetLength(1);

	//	var result = new T[rows * columns];
	float* result = (float*)malloc(width * height * sizeof(float));
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			result[i * width + j] = arr[i, j];
		}
	}
	return result;
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
	//cudaMemcpy(source, image.cudaPtr, image.Width * image.Height, cudaMemcpyDeviceToHost);
	image.GetData(res);
	//for (int i = 0; i < imgHeight * imgWidth; i++)
	//{
	//	printf("%f ", cur[i]);
	//}
	///*float * result = Make1D(cur, imgWidth, imgHeight);
	//return result;*/
}

void main()
{
	int* width = (int*)malloc(sizeof(int));
	int* height = (int*)malloc(sizeof(int));
	char filename[80] = "F:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprinting.GPU.Normalisation\\002.bmp";  //Write your way to bmp file
	int* img = loadBmp(filename, width, height);
	float* source = (float*)malloc((*height)*(*width)*sizeof(float));
	for (int i = 0; i < (*height); i++)
		for (int j = 0; j < (*width); j++)
		{
			source[i * (*width) + j] = (float)img[i, j];
		}
	float* b = (float*)malloc((*height) * (*width) * sizeof(float));
	Normalize(source, b, (*width), (*height), 1000, 1000);
	for (int i = 0; i < (*height) * (*width); i++)
	{
		b[i] = (b[i] < 0) ? 0 : (b[i] > 255 ? 255 : b[i]);
	}
	saveBmp("F:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprinting.GPU.Normalisation\\res.bmp", b, (*width), (*height));
//	int n = 100;
//	float* a = (float*)malloc(n * sizeof(float));
//	for (int i = 0; i < n; i++)
//	{
//		a[i] = i;
//	}
//	float* b = (float*) malloc(n * sizeof(float));
//	b = Normalize(a, 2, 50, 100, 100);
//	for (int i = 0; i < n; i++)
//	{
//		printf("%f ", b[i]);
//	}
	free(width);
	free(height);
	free(img);
	free(b);
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constsmacros.h"
#include "CUDAArray.cuh"
#include <cstring>
#include <time.h>
#include "BinCorrelationHelper.cuh"

void printArray1D(unsigned int* arr, unsigned int length)
{
	for (unsigned int i = 0; i < length; i++) {
		printf("%u ", arr[i]);
	}
	printf("\n");
}

void printCUDAArray1D(CUDAArray<unsigned int> arr)
{
	printf("Print CUDAArray 1D\n");
	printArray1D(arr.GetData(), arr.Width * arr.Height);
	printf("[end] Print CUDAArray 1D\n");
}

void printArray2D(unsigned int* arr, unsigned int width, unsigned int height)
{
	for (unsigned int i = 0; i < height; i++) {
		for (unsigned int j = 0; j < width; j++) {
			printf("%u ", arr[i * width + j]);
		}
		printf("\n");
	}
}

void printCUDAArray2D(CUDAArray<unsigned int> arr)
{
	printf("Print CUDAArray 2D\n");
	printArray2D(arr.GetData(), arr.Width, arr.Height);
	printf("[end] Print CUDAArray 2D\n");
}

__global__ void cudaArrayBitwiseAndGlobal(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd, CUDAArray<unsigned int> result)
{
	if (fst.Width > (defaultColumn()) && fst.Height > (defaultRow())) {
		cudaArrayBitwiseAndDevice(&fst, &snd, result.cudaPtr);
	}
}

CUDAArray<unsigned int> BitwiseAndArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd)
{
	dim3 gridSize = dim3(ceilMod(fst.Width, defaultThreadCount), 1, 1);
	dim3 blockSize = dim3(defaultThreadCount, 1, 1);

	CUDAArray<unsigned int> *result = new CUDAArray<unsigned int>(fst.Width, 1);

	cudaArrayBitwiseAndGlobal << <gridSize, blockSize >> >(fst, snd, *result);

	return *result;
}

__global__ void cudaArrayBitwiseXorGlobal(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd, CUDAArray<unsigned int> result)
{
	if (fst.Width > (defaultColumn()) && fst.Height > (defaultRow())) {
		cudaArrayBitwiseXorDevice(&fst, &snd, result.cudaPtr);
	}
}

CUDAArray<unsigned int> BitwiseXorArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd)
{
	dim3 gridSize = dim3(ceilMod(fst.Width, defaultThreadCount), 1, 1);
	dim3 blockSize = dim3(defaultThreadCount, 1, 1);

	CUDAArray<unsigned int> *result = new CUDAArray<unsigned int>(fst.Width, 1);

	cudaArrayBitwiseXorGlobal << <gridSize, blockSize >> >(fst, snd, *result);

	//printCUDAArray1D(*result);

	return *result;
}

__global__ void cudaArrayWordNormGlobal(CUDAArray<unsigned int> arr, unsigned int* sum)
{
	if (arr.Width > (defaultColumn()) && arr.Height > (defaultRow())) {
		cudaArrayWordNormDevice(arr.cudaPtr, arr.Height, arr.Width, sum);
	}
}

unsigned int getOneBitsCount(CUDAArray<unsigned int> arr)
{
	dim3 gridSize = dim3(ceilMod(arr.Width, defaultThreadCount), 1, 1);
	dim3 blockSize = dim3(defaultThreadCount, 1, 1);

	unsigned int* sum = (unsigned int*)malloc(sizeof(unsigned int));
	*sum = 0;

	unsigned int* d_sum;
	cudaMalloc((unsigned int **)&d_sum, sizeof(unsigned int));
	cudaMemcpy(d_sum, sum, sizeof(unsigned int), cudaMemcpyHostToDevice);

	cudaArrayWordNormGlobal << <gridSize, blockSize >> >(arr, d_sum);

	cudaMemcpy(sum, d_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	return *sum;
}

unsigned int getOneBitsCountRaw(unsigned int* arr, unsigned int length)
{
	CUDAArray<unsigned int> cudaArr(arr, length, 1);
	return getOneBitsCount(cudaArr);
}

void createCylinderValues(char *src, unsigned int srcLength, unsigned int *res)
{
	// srcLength, in symbols/bytes
	// resLength is ints
	unsigned int resLength = ceilMod(srcLength, sizeof(unsigned int)* 8); // 8 obviously for 8 bits per byte

	for (unsigned int i = 0; i < resLength; i++)
	{
		if (srcLength >= (i + 1) * sizeof(unsigned int)* 8)
		{
			char curSrc[sizeof(unsigned int)* 8];
			memcpy(curSrc, &src[i * sizeof(unsigned int)* 8], sizeof(unsigned int)* 8);
			res[i] = strtoul(curSrc, NULL, 2);
		}
		else
		{
			int curSrcLength = srcLength - i * sizeof(unsigned int)* 8 + 1; // + 1 for '\0'
			char *curSrc = (char *)malloc(curSrcLength);
			memcpy(curSrc, &src[i * sizeof(unsigned int)* 8], curSrcLength - 1);
			curSrc[curSrcLength - 1] = '\0';
			int preRes = strtoul(curSrc, NULL, 2);
			res[i] = preRes << (sizeof(unsigned int)* 8 - curSrcLength + 1); // curSrcLength - 1 cause '\0' is not counted
		}
	}
}
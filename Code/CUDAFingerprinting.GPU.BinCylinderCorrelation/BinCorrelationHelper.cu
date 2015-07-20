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

__device__ void cudaArrayBitwiseAndDevice(CUDAArray<unsigned int> *fst, CUDAArray<unsigned int> *snd, CUDAArray<unsigned int> *result)
{
	int row = (defaultRow()) % fst->Height;
	int column = (defaultColumn()) % fst->Width;

	if (fst->Width > column && fst->Height > row) {
		unsigned int newValue = fst->At(row, column) & snd->At(row, column);
		result->SetAt(row, column, newValue);
	}
}

__global__ void cudaArrayBitwiseAndGlobal(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd, CUDAArray<unsigned int> result)
{
	if (fst.Width > (defaultColumn()) && fst.Height > (defaultRow())) {
		cudaArrayBitwiseAndDevice(&fst, &snd, &result);
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


__device__ void cudaArrayBitwiseXorDevice(CUDAArray<unsigned int> *fst, CUDAArray<unsigned int> *snd, CUDAArray<unsigned int> *result)
{
	int row = (defaultRow()) % fst->Height;
	int column = (defaultColumn()) % fst->Width;

	unsigned int newValue = fst->At(row, column) ^ snd->At(row, column);
	result->SetAt(row, column, newValue);
}

__global__ void cudaArrayBitwiseXorGlobal(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd, CUDAArray<unsigned int> result)
{
	if (fst.Width > (defaultColumn()) && fst.Height > (defaultRow())) {
		cudaArrayBitwiseXorDevice(&fst, &snd, &result);
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

__device__ void cudaArrayWordNormDevice(CUDAArray<unsigned int> *arr, unsigned int* sum)
{
	int row = (defaultRow()) % arr->Height;
	int column = (defaultColumn()) % arr->Width;

	unsigned int x = arr->At(row, column);

	x = __popc(x);

	atomicAdd(sum, x);
}

__global__ void cudaArrayWordNormGlobal(CUDAArray<unsigned int> arr, unsigned int* sum)
{
	if (arr.Width > (defaultColumn()) && arr.Height > (defaultRow())) {
		cudaArrayWordNormDevice(&arr, sum);
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

unsigned int binToInt(char* s)
{
	return (unsigned int)strtoul(s, NULL, 2);
}
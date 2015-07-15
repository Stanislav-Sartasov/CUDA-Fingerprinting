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
	printf("Print CUDAArray\n");
	printArray1D(arr.GetData(), arr.Width * arr.Height);
	printf("[end] Print CUDAArray\n");
}

__global__ void cudaArrayBitwiseAnd(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd, CUDAArray<unsigned int> result)
{
	int row = defaultRow();
	int column = defaultColumn();

	if (fst.Width > column && fst.Height > row) {
		unsigned int newValue = fst.At(row, column) & snd.At(row, column);
		result.SetAt(row, column, newValue);
	}
}

CUDAArray<unsigned int> BitwiseAndArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd)
{
	dim3 gridSize = dim3(ceilMod(fst.Width, defaultThreadCount), 1, 1);
	dim3 blockSize = dim3(defaultThreadCount, 1, 1);

	CUDAArray<unsigned int> *result = new CUDAArray<unsigned int>(fst.Width, 1);

	cudaArrayBitwiseAnd << <gridSize, blockSize >> >(fst, snd, *result);

	return *result;
}


__global__ void cudaArrayBitwiseXor(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd, CUDAArray<unsigned int> result)
{
	int row = defaultRow();
	int column = defaultColumn();

	if (fst.Width > column && fst.Height > row)	{
		unsigned int newValue = fst.At(row, column) ^ snd.At(row, column);
		result.SetAt(row, column, newValue);
	}
}

CUDAArray<unsigned int> BitwiseXorArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd)
{
	dim3 gridSize = dim3(ceilMod(fst.Width, defaultThreadCount), 1, 1);
	dim3 blockSize = dim3(defaultThreadCount, 1, 1);

	CUDAArray<unsigned int> *result = new CUDAArray<unsigned int>(fst.Width, 1);

	cudaArrayBitwiseXor << <gridSize, blockSize >> >(fst, snd, *result);

	//printCUDAArray1D(*result);

	return *result;
}

__global__ void cudaArrayWordNorm(CUDAArray<unsigned int> arr, unsigned int* sum)
{
	int row = defaultRow();
	int column = defaultColumn();

	if (arr.Width > column && arr.Height > row)	{
		unsigned int x = arr.At(row, column);

		x = x - ((x >> 1) & 0x55555555);
		x = (x & 0x33333333) + ((x >> 2) & 0x33333333);
		x = (x + (x >> 4)) & 0x0F0F0F0F;
		x = x + (x >> 8);
		x = x + (x >> 16);
		x = x & 0x0000003F;

		atomicAdd(sum, x);
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

	cudaArrayWordNorm << <gridSize, blockSize >> >(arr, d_sum);

	cudaMemcpy(sum, d_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	return *sum;
}


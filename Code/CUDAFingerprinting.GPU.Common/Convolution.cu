#include "Convolution.cuh"
#include "constsmacros.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <float.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef __CUDACC__ 
#define __CUDACC__
#endif
#include <device_functions.h>

// GPU FUNCTIONS

__global__ void cudaArrayAdd(CUDAArray<float> source, CUDAArray<float> addition)
{
	int row = defaultRow();
	int column = defaultColumn();
	if (source.Width>column&&source.Height>row)
	{
		float newValue = source.At(row, column) + addition.At(row, column);
		source.SetAt(row, column, newValue);
	}
}

__global__ void cudaArraySubtract(CUDAArray<float> source, CUDAArray<float> subtract)
{
	int row = defaultRow();
	int column = defaultColumn();
	if (source.Width>column&&source.Height>row)
	{
		float newValue = source.At(row, column) - subtract.At(row, column);
		source.SetAt(row, column, newValue);
	}
}

__global__ void cudaConvolve(CUDAArray<float> target, CUDAArray<float> source, CUDAArray<float> filter)
{
	int row = defaultRow();
	int column = defaultColumn();
	if (source.Width>column&&source.Height>row)
	{
		int tX = threadIdx.x;
		int tY = threadIdx.y;
		__shared__ float filterCache[32 * 32];

		if (tX<filter.Width&&tY<filter.Height)
		{
			int indexLocal = tX + tY*filter.Width;
			filterCache[indexLocal] = filter.At(tY, tX);
		}
		__syncthreads();

		int center = filter.Width / 2;

		float sum = 0.0f;

		for (int drow = -center; drow <= center; drow++)
		{
			for (int dcolumn = -center; dcolumn <= center; dcolumn++)
			{
				float filterValue1 = filterCache[filter.Width*(drow + center) + dcolumn + center];

				int valueRow = row + drow;
				int valueColumn = column + dcolumn;

				if (valueRow<0 || valueRow >= source.Height || valueColumn<0 || valueColumn >= source.Width)
					continue;

				float value = source.At(valueRow, valueColumn);
				sum += filterValue1*value;
			}
		}

		target.SetAt(row, column, sum);
	}
}

// CPU FUNCTIONS

void AddArray(CUDAArray<float> source, CUDAArray<float> addition)
{
	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize =
		dim3(ceilMod(source.Width, defaultThreadCount),
		ceilMod(source.Height, defaultThreadCount));

	cudaArrayAdd<<<gridSize, blockSize >> >(source, addition);
}

void SubtractArray(CUDAArray<float> source, CUDAArray<float> subtract)
{
	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize =
		dim3(ceilMod(source.Width, defaultThreadCount),
		ceilMod(source.Height, defaultThreadCount));

	cudaArraySubtract<<<gridSize, blockSize>>>(source, subtract);
}

void Convolve(CUDAArray<float> target, CUDAArray<float> source, CUDAArray<float> filter)
{
	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize =
		dim3(ceilMod(source.Width, defaultThreadCount),
		ceilMod(source.Height, defaultThreadCount));

	cudaConvolve<<<gridSize, blockSize>>>(target, source, filter);
	cudaError_t error = cudaDeviceSynchronize();
}

void ComplexConvolve(CUDAArray<float> targetReal, CUDAArray<float> targetImaginary,
	CUDAArray<float> sourceReal, CUDAArray<float> sourceImaginary,
	CUDAArray<float> filterReal, CUDAArray<float> filterImaginary)
{
	CUDAArray<float> tempReal = CUDAArray<float>(targetReal.Width, targetReal.Height);
	CUDAArray<float> tempImaginary = CUDAArray<float>(targetImaginary.Width, targetImaginary.Height);

	Convolve(targetReal, sourceReal, filterReal);
	Convolve(tempReal, sourceImaginary, filterImaginary);

	Convolve(targetImaginary, sourceReal, filterImaginary);
	Convolve(tempImaginary, sourceImaginary, filterReal);

	AddArray(targetImaginary, tempImaginary);
	SubtractArray(targetReal, tempReal);

	tempReal.Dispose();
	tempImaginary.Dispose();
}
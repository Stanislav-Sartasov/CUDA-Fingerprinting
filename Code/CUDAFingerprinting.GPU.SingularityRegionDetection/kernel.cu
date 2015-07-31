#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_constants.h>

#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
#include "Convolution.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

const int defaultBlockSize = 16;
const int bigBlockSize = 48;

__global__ void cudaSinCos (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> source)
{
	int row = defaultRow ();
	int column = defaultColumn ();
	if (row < cMapReal.Height && column < cMapReal.Width && row >= 0 && column >= 0)
	{
		float value = source.At(row, column);
		float sinValue = sin(2 * value);
		float cosValue = cos(2 * value);
		cMapImaginary.SetAt(row, column, cosValue);
		cMapReal.SetAt(row, column, sinValue);
	}	
}

void SinCos (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> source)
{
	dim3 blockSize = dim3 (defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3 (ceilMod (cMapReal.Width, defaultBlockSize), ceilMod (cMapReal.Height, defaultBlockSize));

	cudaSinCos <<< gridSize, blockSize >>> ( cMapReal, cMapImaginary, source );
}

__global__ void cudaModule (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> cMapAbs)
{
	int row = defaultRow ();
	int column = defaultColumn ();

	if (row < cMapReal.Height && column < cMapReal.Width)
	{
		int tX = threadIdx.x;
		int tY = threadIdx.y;

		__shared__ float bufReal[defaultBlockSize][defaultBlockSize];
		__shared__ float bufImaginary[defaultBlockSize][defaultBlockSize];
		__shared__ float bufAbs[defaultBlockSize][defaultBlockSize];

		bufReal[tX][tY] = cMapReal.At(row, column);
		bufImaginary[tX][tY] = cMapImaginary.At(row, column);

		__syncthreads();
		if (tX == 0 && tY == 0)
		{
			for (int i = 0; i < 16; i++)
			{
				for (int j = 0; j < 16; j++)
				{
					float R = bufReal[i][j];
					float I = bufImaginary[i][j];
					bufAbs[i][j] = sqrt(R * R + I * I);
				}
			}
		}
		__syncthreads();

		cMapAbs.SetAt(row, column, bufAbs[tX][tY]);
	}
}

void Module (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> cMapAbs)
{
	dim3 blockSize = dim3 (defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3 (ceilMod (cMapReal.Width, defaultBlockSize), ceilMod (cMapReal.Height, defaultBlockSize));

	cudaModule <<< gridSize, blockSize >>> (cMapReal, cMapImaginary, cMapAbs);
}

__global__ void cudaRegularize (CUDAArray<float> source, CUDAArray<float> target)
{
	int row = defaultRow();
	int column = defaultColumn();

	if (row >= 8 && row < source.Height && column >= 8 && column < source.Width)
	{
		int tX = threadIdx.x;
		int tY = threadIdx.y;

		__shared__ float buf[defaultBlockSize][defaultBlockSize];
		__shared__ float linBuf[defaultBlockSize];
		for (int i = -1; i < 2; i++)
		{
			for (int j = -1; j < 2; j++)
			{
				buf[tX + 8 * i][tY + 8 * j] = source.At(row + 8 * i, column + 8 * j);
			}
		}			

		__syncthreads();
		if (tX == 0)
		{
			float sum = 0;
			for (int i = 0; i < defaultBlockSize; i++)
			{
				sum += buf[i][tY];
			}
			linBuf[tY] = sum;
		}

		__syncthreads();
		if (tX == 0 && tY == 0)
		{
			float sum = 0;
			for (int i = 0; i < defaultBlockSize; i++)
			{
				sum += linBuf[i];
			}
			linBuf[0] = sum;
		}
		__syncthreads();

		float val = linBuf[0] / (defaultBlockSize * defaultBlockSize);

		target.SetAt(row, column, val);
	}
}

void Regularize (CUDAArray<float> source, CUDAArray <float> target)
{
	dim3 blockSize = dim3 (defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3 (ceilMod (source.Width, defaultBlockSize), ceilMod (source.Height, defaultBlockSize));

	cudaRegularize <<< gridSize, blockSize >>> ( source, target );
}

__device__ __inline__ float bufValue (CUDAArray<float> source, int row, int column)
{
	if (row - 3 >= 0 && row + 3 < source.Height && column - 3 >= 0 && column + 3 < source.Width)
	{
		return source.At (row, column);
	}
	else
	{
		return 0;
	}
}

__device__ __inline__ float Sum (CUDAArray<float> source, int tX, int tY, int row, int column)
{
	__shared__ float buf[9][9];

	/*for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++)
		{
			float val = bufValue(source, row + 3 * i, column + 3 * j);
			buf[tX][tY] = val;
		}
	}*/

	buf[tX][tY] = source.At(row, column);

	__syncthreads();	
	float sum = 0;
	for ( int i = 0; i < 9; i++ )
	{
		for ( int j = 0; j < 9; j++ )
		{
			sum += buf[i][j];
		}			
	}
	__syncthreads();

	return sum;
}

__global__ void cudaStrengthen (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> cMapAbs, CUDAArray<float> target)
{
	int row = defaultRow();
	int column = defaultColumn();

	if (row < cMapReal.Height && column < cMapReal.Width)
	{
		int tX = threadIdx.x;
		int tY = threadIdx.y;

		float R = Sum(cMapReal, tX, tY, row, column);
		float I = Sum(cMapImaginary, tX, tY, row, column);

		float numerator = sqrt(R * R + I * I);
		float denominator = Sum(cMapAbs, tX, tY, row, column);

		float val = 1 - numerator / denominator;

		target.SetAt(row, column, val);
	}
}

void Strengthen (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> cMapAbs, CUDAArray<float> result)
{
	dim3 blockSize = dim3(9, 9);
	dim3 gridSize = dim3(ceilMod(cMapReal.Width, 9), ceilMod(cMapReal.Height, 9));

	cudaStrengthen <<< gridSize, blockSize >>> (cMapReal, cMapImaginary, cMapAbs, result);
	cudaError_t error = cudaGetLastError ();
}

void SaveSegmentation (int width, int height, float* matrix)
{
	int* newPic = (int*) malloc (sizeof (int)*width*height);
	int capacity = width * height;

	for ( int i = 0; i < capacity; i++ )
	{
		newPic[i] = (int)(matrix[i] * 255);
	}

	saveBmp ("Result.jpg", newPic, width, height);
	
	std::free (newPic);
}

void Detect (float* orient, int width, int height)
{
	CUDAArray<float> source = CUDAArray<float> (orient, width, height);
	CUDAArray<float> result = CUDAArray<float> (width, height);

	CUDAArray<float> cMapReal = CUDAArray<float> (width, height);
	CUDAArray<float> cMapImaginary = CUDAArray<float> (width, height);
	CUDAArray<float> cMapAbs = CUDAArray<float> (width, height);

	SinCos (cMapReal, cMapImaginary, source);
	cudaError_t error = cudaGetLastError();

	Module(cMapReal, cMapImaginary, cMapAbs);

	CUDAArray<float> V_rReal = CUDAArray<float> (width, height);
	CUDAArray<float> V_rImaginary = CUDAArray<float> (width, height);

	//Regularize ( V_rReal, cMapReal );
	//Regularize ( V_rImaginary, cMapImaginary );

	Strengthen (cMapReal, cMapImaginary, cMapAbs, result);
	float* str = new float[width * height];
	result.GetData (str);
	SaveSegmentation (width, height, str);

	std::free (str);

	source.Dispose();
	result.Dispose();
	cMapReal.Dispose();
	cMapImaginary.Dispose();
	cMapAbs.Dispose();
	V_rReal.Dispose();
	V_rImaginary.Dispose();
}

int main()
{
	cudaSetDevice (0);

	int width, height;
	int* pic = loadBmp ("1_1.bmp", &width, &height);

	/*float* fPic  = (float*) malloc (sizeof (float)*width*height);
	for ( int i = 0; i < width * height; i++ )
	{
		fPic[i] = (float) pic[i];
	}*/

	float* orient = new float [width * height];

	ifstream fin ("Orientation.txt");
	float value = 0;
	for ( int i = 0; i < height; i++ )
	{
		for ( int j = 0; j < width; j++ )
		{
			fin >> value;
			orient[j + i * width] = value / 10000;
		}
	}

	//OrientationFieldInPixels (orient, fPic, width, height);

	Detect ( orient, width, height );

	std::free(pic);
	//free (fPic);
	std::free (orient);

	return 0;
}
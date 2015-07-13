#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
#include "Convolution.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>

using namespace std;

#define edge 100
#define defaultBlockSize 16

extern "C"
{
	__declspec( dllexport ) __global__ void cudaSegmentate (CUDAArray<float> value, int* matrix);
}

__global__ void cudaCalculate(CUDAArray<float> value, CUDAArray<float> Gx, CUDAArray<float> Gy)
{
	int row = defaultRow();
	int column = defaultColumn();

	int width = value.Width;
	int height = value.Height;

	float currentGx = Gx.At(row, column);
	float currentGy = Gy.At(row, column);

	if( column < width && row < height )
	{
		float sqrtXY = sqrt( currentGx * currentGx + currentGy * currentGy );
		value.SetAt(row, column, sqrtXY);
	}
}

void Calculate(CUDAArray<float> value, CUDAArray<float> Gx, CUDAArray<float> Gy)
{
	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3(ceilMod(value.Width, defaultBlockSize), ceilMod(value.Height, defaultBlockSize));

	cudaCalculate<<< gridSize, blockSize >>>(value, Gx, Gy);
}

CUDAArray<float> SobelFilter (CUDAArray<float> source, int picWidth, int picHeight)
{
	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	CUDAArray<float> filterX(filterXLinear, 3, 3);
	CUDAArray<float> filterY(filterYLinear, 3, 3);
	
	CUDAArray<float> Gx(picWidth, picHeight);
	CUDAArray<float> Gy(picWidth, picHeight);

	Convolve(Gx, source, filterX);
	Convolve(Gy, source, filterY);

	CUDAArray<float> value = CUDAArray<float>(picWidth, picHeight);
	Calculate(value, Gx, Gy); // Calculates result of Sobel Operator

	filterX.Dispose();
	filterY.Dispose();
	Gx.Dispose();
	Gy.Dispose();

	return value;
}

__global__ void cudaMatrix (CUDAArray<float> value, CUDAArray<int> matrix2D)
{
	int row = defaultRow ();
	int column = defaultColumn ();

	__shared__ float buf[16][16];
	buf[threadIdx.x][threadIdx.y] = value.At (row, column);

	__syncthreads();
	if ( threadIdx.x == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; ++i )
		{
			sum += buf[i][threadIdx.y];
		}
		buf[0][threadIdx.y] = sum;
	}

	__syncthreads();
	if ( threadIdx.x == 0 && threadIdx.y == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; ++i )
		{
			sum += buf[0][i];
		}
		buf[0][0] = sum;
	}
	__syncthreads();

	float val = buf[0][0] / ( defaultBlockSize * defaultBlockSize );

	if ( val >= edge )
	{
		matrix2D.SetAt (row, column, 1);
	}
	else
	{
		matrix2D.SetAt (row, column, 0);
	}

	/*if ( column < width && row < height )
	{
		int columnOffset = defaultBlockSize * column;
		int rowOffset = defaultBlockSize * row;

		float averageColor = 0;

		for( int i = 0; i < defaultBlockSize; ++i )
		{
			if( columnOffset + i < width )
			{
				for( int j = 0; j < defaultBlockSize; ++j )
				{
					if ( rowOffset + j < height )
					{
						averageColor += value.At( rowOffset + j, columnOffset + i );
					}		
				}
			}			
		}

		averageColor /= (defaultBlockSize * defaultBlockSize);

		if ( averageColor >= edge )
		{
			matrix2D.SetAt (defaultRow (), defaultColumn (), 1);
		}
		else
		{
			matrix2D.SetAt (defaultRow (), defaultColumn (), 0);
		}
	}*/
}

void Segmentate (CUDAArray<float> value, int* matrix)
{
	int width = value.Width;
	int height = value.Height;

	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3(ceilMod(value.Width, defaultBlockSize), ceilMod(value.Height, defaultBlockSize));

	CUDAArray<int> matrix2D = CUDAArray<int>(matrix, width, height);
	cudaMatrix << < gridSize, blockSize >> >(value, matrix2D);

	matrix2D.GetData(matrix);

	matrix2D.Dispose();
}

void BWPicture (int width, int height, int* matrix)
{
	int* newPic;
	int capacity = width * height;

	for ( int i = 0; i < capacity; ++i )
	{
		newPic[i] = matrix[i] * 255;
	}

	saveBmp ("newPic.bmp", newPic, width, height);
}

void MakingMatrix (float* fPic, int picWidth, int picHeight, int* &matrix)
{
	CUDAArray<float> source = CUDAArray<float>(fPic, picWidth, picHeight);

	CUDAArray<float> value = SobelFilter (source, picWidth, picHeight);	
	Segmentate (value, matrix);
	
	BWPicture (picWidth, picHeight, matrix);

	source.Dispose();
	value.Dispose ();
}

int main()
{
	cudaSetDevice (0);

	int picWidth, picHeight;
	int* pic = loadBmp ("1_1.bmp", &picWidth, &picHeight);
	float* fPic = (float*)malloc(sizeof(float)*picWidth*picHeight);
	for ( int i = 0; i < picWidth*picHeight; i++ )
	{
		fPic[i] = (float) pic[i];
	}

	int *matrix = (int*) malloc (picWidth * picHeight * sizeof(int));
	// In this matrix 1 means light shade of gray, and 0 means dark shade of gray 

	MakingMatrix (fPic, picWidth, picHeight, matrix);

	free(pic);
	free (fPic);
	free (matrix);

	return 0;
}
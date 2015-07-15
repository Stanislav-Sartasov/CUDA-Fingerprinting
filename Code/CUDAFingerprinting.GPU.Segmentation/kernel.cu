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

#define edge 50
#define pixEdge 150
#define defaultBlockSize 16

extern "C"
{
	__declspec( dllexport ) void Segmentate (CUDAArray<float> value, int* matrix);
}

__global__ void SobelFilter (CUDAArray<float> source, CUDAArray<float> target, CUDAArray<float> filterX, CUDAArray<float> filterY)
{
	int row = defaultRow ();
	int column = defaultColumn ();

	if (column < source.Width - 1 && row < source.Height - 1 && column > 0 && row > 0 )
	{
        float sumX = source.At(row - 1, column - 1) * filterX.At(0, 0) + source.At(row + 1, column - 1) * filterX.At(0, 2) +
                        source.At(row - 1, column) * filterX.At(1, 0) + source.At(row + 1, column) * filterX.At(1, 2) +
                        source.At(row - 1, column + 1) * filterX.At(2, 0) + source.At(row + 1, column + 1) * filterX.At(2, 2);
        
		float sumY = source.At(row - 1, column - 1) * filterY.At(0, 0) + source.At(row, column - 1) * filterY.At(0, 1) + source.At(row + 1, column - 1) * filterY.At(0, 2) +
            source.At(row - 1, column + 1) * filterY.At(2, 0) + source.At(row, column + 1) * filterY.At(2, 1) + source.At(row + 1, column + 1) * filterY.At(2, 2);
        
		float sqrtXY = sqrt(sumX * sumX + sumY * sumY);

        sqrtXY = sqrtXY > 255 ? 255 : sqrtXY;

		target.SetAt(row, column, sqrtXY);
    }
}

__global__ void cudaMatrix (CUDAArray<float> value, CUDAArray<int> matrix2D)
{
	int row = blockIdx.y * defaultBlockSize + threadIdx.y;
	int column = blockIdx.x * defaultBlockSize + threadIdx.x;

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	__shared__ float buf[defaultBlockSize][defaultBlockSize];
	__shared__ float linBuf[defaultBlockSize];
	buf[tX][tY] = value.At (row, column);

	__syncthreads();
	if ( tX == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; ++i )
		{
			sum += buf[i][tY];
		}
		linBuf[tY] = sum;
	}

	__syncthreads();
	if ( tX == 0 && tY == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; ++i )
		{
			sum += linBuf[i];
		}
		linBuf[0] = sum;
	}
	__syncthreads();

	float val = linBuf[0] / ( defaultBlockSize * defaultBlockSize );

	if ( val >= edge )
	{
		if ( buf[tX][tY] >= pixEdge )
		{
			matrix2D.SetAt (row, column, 1);
		}
		else
		{
			matrix2D.SetAt (row, column, 0);
		}
	}
	else
	{
		matrix2D.SetAt (row, column, 0);
	}
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
	int* newPic = (int*) malloc (sizeof (int)*width*height);
	int capacity = width * height;

	for ( int i = 0; i < capacity; ++i )
	{
		newPic[i] = matrix[i] * 255;
	}

	saveBmp ("newPic.bmp", newPic, width, height);

	free (newPic);
}

void MakingMatrix (float* fPic, int picWidth, int picHeight, int* matrix)
{
	CUDAArray<float> source = CUDAArray<float>(fPic, picWidth, picHeight);
	CUDAArray<float> target = CUDAArray<float>(picWidth, picHeight);

	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	CUDAArray<float> filterX(filterXLinear, 3, 3);
	CUDAArray<float> filterY(filterYLinear, 3, 3);

	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3(ceilMod(picWidth, defaultBlockSize), ceilMod(picHeight, defaultBlockSize));

	SobelFilter <<< gridSize, blockSize >>> (source, target, filterX, filterY);	
	
	//Saving image after Sobel Filter
	float* fSOPic = (float*)malloc(sizeof(float)*source.Width*source.Height);
	target.GetData (fSOPic);
	int* SOPic = (int*)malloc(sizeof(int)*source.Width*source.Height);
	for ( int i = 0; i < picWidth*picHeight; i++ )
	{
		SOPic[i] = (int) fSOPic[i];
	}
	saveBmp ("SOPic.bmp", SOPic, picWidth, picHeight);
	
	Segmentate (target, matrix);
	BWPicture (picWidth, picHeight, matrix);

	source.Dispose();
	target.Dispose();
	filterX.Dispose();
	filterY.Dispose();

	free (fSOPic);
	free (SOPic);
}

int main()
{
	cudaSetDevice (0);

	int picWidth, picHeight;
	int* pic = loadBmp ("..//1_1.bmp", &picWidth, &picHeight);
	float* fPic  = (float*) malloc (sizeof (float)*picWidth*picHeight);
	for ( int i = 0; i < picWidth * picHeight; i++ )
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
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

#define edge 80
#define defaultBlockSize 16

extern "C"
{
	__declspec( dllexport ) __global__ void cudaSegmentate (CUDAArray<float> value, int* matrix);
}

/*__global__ void cudaCalculate(CUDAArray<float> value, CUDAArray<float> Gx, CUDAArray<float> Gy)
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
}*/


CUDAArray<float> SobelFilter (CUDAArray<float> source, int picWidth, int picHeight)
{
	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	CUDAArray<float> cudaFilterX(filterXLinear, 3, 3);
	CUDAArray<float> cudaFilterY(filterYLinear, 3, 3);

	float* filterX = (float*) malloc (sizeof (float)*3*3);
	cudaFilterX.GetData (filterX);

	float* filterY = (float*) malloc (sizeof (float)*3*3);
	cudaFilterY.GetData (filterY);

	float* fSource = (float*) malloc (sizeof (float)*picWidth*picHeight);
	source.GetData (fSource);
	float* fRes = (float*) malloc (sizeof (float)*picWidth*picHeight);

	for (int x = 1; x < picWidth - 1; x++)
    {
        for (int y = 1; y < picHeight - 1; y++)
        {
            float sumX = fSource[x - 1 + picWidth * (y - 1)] * filterX[0] + fSource[x + 1 + picWidth * (y - 1)] * filterX[2] +
                            fSource[x - 1 + picWidth * y] * filterX[3] + fSource[x + 1 + picWidth * y] * filterX[5] +
                            fSource[x - 1 + picWidth * (y + 1)] * filterX[6] + fSource[x + 1 + picWidth * (y + 1)] * filterX[8];
            float sumY = fSource[x - 1 + picWidth * (y - 1)] * filterY[0] + fSource[x + picWidth * (y - 1)] * filterY[1] + fSource[x + 1 + picWidth * (y - 1)] * filterY[2] +
                fSource[x - 1 + picWidth * (y + 1)] * filterY[6] + fSource[x + picWidth * (y + 1)] * filterY[7] + fSource[x + 1 + picWidth * (y + 1)] * filterY[8];
            double sqrtXY = sqrt(sumX * sumX + sumY * sumY);

            sqrtXY = sqrtXY > 255 ? 255 : sqrtXY;

			fRes [x + picWidth * y] = sqrtXY;
        }
    }

	CUDAArray<float> value = CUDAArray<float>(fRes, picWidth, picHeight);

	/*int* SOPic = (int*)malloc(sizeof(int)*value.Width*value.Height);
	for ( int i = 0; i < picWidth*picHeight; i++ )
	{
		SOPic[i] = (int) fRes[i];
	}
	saveBmp ("SOPic.bmp", SOPic, picWidth, picHeight);*/

	cudaFilterX.Dispose();
	cudaFilterY.Dispose();

	return value;
}

__global__ void cudaMatrix (CUDAArray<float> value, CUDAArray<int> matrix2D)
{
	int row = blockIdx.y * defaultBlockSize + threadIdx.y;
	int column = blockIdx.x * defaultBlockSize + threadIdx.x;

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	__shared__ float buf[defaultBlockSize][defaultBlockSize];
	buf[tX][tY] = value.At (row, column);

	__syncthreads();
	if ( tX == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; ++i )
		{
			sum += buf[i][tY];
		}
		buf[0][tY] = sum;
	}

	__syncthreads();
	if ( tX == 0 && tY == 0 )
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
		if ( buf[tX][tY] >= edge )
			matrix2D.SetAt (row, column, 1);
		else
			matrix2D.SetAt (row, column, 0);
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

	ofstream f;
	f.open ("matrix.txt");
	for ( int i = 0; i < value.Width; ++i )
	{
		for ( int j = 0; j < value.Height; ++j )
		{
			f << matrix[i * value.Width + j] << ' ';
		}
		f << endl;
	}

	matrix2D.Dispose();
}

void BWPicture (int width, int height, int* matrix)
{
	int* newPic = (int*) malloc (sizeof (int)*width*height);;
	int capacity = width * height;

	for ( int i = 0; i < capacity; ++i )
	{
		newPic[i] = matrix[i] * 255;
	}

	saveBmp ("newPic.bmp", newPic, width, height);
}

void MakingMatrix (float* fPic, int picWidth, int picHeight, int* matrix)
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
	int* pic = loadBmp ("..//1_1.bmp", &picWidth, &picHeight);
	float* fPic  = (float*) malloc (sizeof (float)*picWidth*picHeight);
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
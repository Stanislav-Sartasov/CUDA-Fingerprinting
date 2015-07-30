#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_constants.h>

#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
#include "Convolution.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>

using namespace std;

const int defaultBlockSize = 16;
const int bigBlockSize = 48;

__global__ void cudaSinCos (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> source)
{
	int row = defaultRow ();
	int column = defaultColumn ();

	/*int tX = threadIdx.x;
	int tY = threadIdx.y;
	
	__shared__ float bufSource[defaultBlockSize][defaultBlockSize];
	__shared__ float bufReal[defaultBlockSize][defaultBlockSize];
	__shared__ float bufImaginary[defaultBlockSize][defaultBlockSize];

	
	bufSource[tX][tY] = source.At(row, column);

	__syncthreads ();
	if ( tX == 0 && tY == 0 )
	{
		for ( int i = 0; i < source.Width; i++ )
		{
			for ( int j = 0; j < source.Height; j++ )
			{
				bufReal[i][j] = sin (2 * bufSource[i][j]);
				bufImaginary[i][j] = cos (2 * bufSource[i][j]);
			}
		}
	}
	__syncthreads ();
	
	cMapReal.SetAt (row, column, bufReal[tX][tY]);
	cMapImaginary.SetAt (row, column, bufImaginary[tX][tY]);*/

	cMapReal.SetAt(row, column, sin(2 * source.At(row, column)));
	cMapImaginary.SetAt(row, column, cos(2 * source.At(row, column)));
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

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	__shared__ float bufReal[defaultBlockSize][defaultBlockSize];
	__shared__ float bufImaginary[defaultBlockSize][defaultBlockSize];
	__shared__ float bufAbs[defaultBlockSize][defaultBlockSize];

	bufReal[tX][tY] = cMapReal.At(row, column);
	bufImaginary[tX][tY] =  cMapImaginary.At (row, column);

	__syncthreads ();
	if ( tX == 0 && tY == 0 )
	{
		for ( int i = 0; i < cMapReal.Width; i++ )
		{
			for ( int j = 0; j < cMapReal.Height; j++ )
			{
				float R = bufReal[i][j];
				float I = bufImaginary[i][j];
				bufAbs[i][j] = sqrt(R * R + I * I);
			}
		}
	}
	__syncthreads ();

	cMapAbs.SetAt (row, column, bufAbs[tX][tY]);
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

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	__shared__ float buf[defaultBlockSize][defaultBlockSize];
	__shared__ float linBuf[defaultBlockSize];
	buf[tX][tY] = source.At (row, column);

	__syncthreads();
	if ( tX == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; i++ )
		{
			sum += buf[i][tY];
		}
		linBuf[tY] = sum;
	}

	__syncthreads();
	if ( tX == 0 && tY == 0 )
	{
		float sum = 0;
		for ( int i = 0; i < defaultBlockSize; i++ )
		{
			sum += linBuf[i];
		}
		linBuf[0] = sum;
	}
	__syncthreads();

	float val = linBuf[0] / ( defaultBlockSize * defaultBlockSize );

	target.SetAt (row, column, val);
}

void Regularize (CUDAArray<float> source, CUDAArray <float> target)
{
	dim3 blockSize = dim3 (defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3 (ceilMod (source.Width, defaultBlockSize), ceilMod (source.Height, defaultBlockSize));

	cudaRegularize <<< gridSize, blockSize >>> ( source, target );
}

__device__ __inline__ float bufValue (CUDAArray<float> source, int row, int column)
{
	if ( ( row - 16 >= 0 ) && ( row + 16 < source.Height ) && ( column - 16 >= 0 ) && ( column + 16 < source.Width ) )
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
	__shared__ float buf[48][48];

	if ( tX >= 16 && tX <= 31 && tY >= 16 && tY <= 31 )
		for ( int i = -1; i < 2; i++ )
			for ( int j = -1; j < 2; j++ )
				buf[tX + 16 * i][tY + 16 * j] = bufValue (source, row + 16 * i, column + 16 * j);

	__syncthreads();
	
	if ( tX == 0 && tY == 0 )
	{
		float sum = 0;
		for ( int i = tX - 16; i < tX + 16; i++ )
		{
			for ( int j = tY - 16; j < tY + 16; j++ )
			{
				sum += buf[i][j];
			}			
		}
		buf[0][0] = sum;
	}
	__syncthreads();

	return buf[0][0];
}

__global__ void cudaStrengthen (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> cMapAbs, CUDAArray<float> target)
{
	int row = defaultRow();
	int column = defaultColumn();

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	float R = Sum (cMapReal, tX, tY, row, column);
	float I = Sum (cMapImaginary, tX, tY, row, column);

	float numerator = sqrt (R * R + I * I);
	float denominator = Sum (cMapAbs, tX, tY, row, column);

	float val = 1 - numerator / denominator;

	target.SetAt (row, column, val);
}

void Strengthen (CUDAArray<float> cMapReal, CUDAArray<float> cMapImaginary, CUDAArray<float> cMapAbs, CUDAArray<float> result)
{
	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3(ceilMod(cMapReal.Width, defaultBlockSize), ceilMod(cMapReal.Height, defaultBlockSize));

	cudaStrengthen <<< gridSize, blockSize >>> (cMapReal, cMapImaginary, cMapAbs, result);
	cudaError_t error = cudaGetLastError ();
}

void SaveSegmentation (int width, int height, float* matrix)
{
	int* newPic = (int*) malloc (sizeof (int)*width*height);
	int capacity = width * height;

	for ( int i = 0; i < capacity; ++i )
	{
		newPic[i] = (int)(matrix[i] * 255);
	}

	saveBmp ("Result.jpg", newPic, width, height);
	
	free (newPic);
}

void Detect (float* orient, int width, int height)
{
	CUDAArray<float> source = CUDAArray<float> (orient, width, height);
	CUDAArray<float> result = CUDAArray<float> (width, height);

	CUDAArray<float> cMapReal = CUDAArray<float> (width, height);
	CUDAArray<float> cMapImaginary = CUDAArray<float> (width, height);
	CUDAArray<float> cMapAbs = CUDAArray<float> (width, height);

	float* ptr1 = new float[width * height];
	float* ptr2 = new float[width * height];

	SinCos (cMapReal, cMapImaginary, source);
	cMapReal.GetData (ptr1);
	ptr2 = cMapImaginary.GetData ();

	ofstream sin ("gpuSin.doc");
	for ( int i = 0; i < width; i++ )
	{
		sin << i << "] ";
		for 
		( int j = 0; j < height; j++ )
		{
			sin << j << ") " << ptr1[i*width + j] << " ";
		}
		sin << endl;
	}
	sin.close();

	ofstream cos("gpuCos.doc");
	for (int i = 0; i < width; i++)
	{
		cos << i << "] ";
		for
			(int j = 0; j < height; j++)
		{
			cos << j << ") " << ptr2[i * width + j] << " ";
		}
		cos << endl;
	}
	cos.close();

	Module (cMapReal, cMapImaginary, cMapAbs);

	CUDAArray<float> V_rReal = CUDAArray<float> (width, height);
	CUDAArray<float> V_rImaginary = CUDAArray<float> (width, height);

	Regularize ( V_rReal, cMapReal );
	Regularize ( V_rImaginary, cMapImaginary );

	Strengthen (cMapReal, cMapImaginary, cMapAbs, result);
	float* str = new float[width * height];
	result.GetData (str);
	SaveSegmentation (width, height, str);

	free (str);

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
	for ( int i = 0; i < width; i++ )
	{
		for ( int j = 0; j < height; j++ )
		{
			fin >> value;
			orient[i + j * width] = value / 10000;
		}
	}

	//OrientationFieldInPixels (orient, fPic, width, height);

	Detect ( orient, width, height );

	free(pic);
	//free (fPic);
	free (orient);

	return 0;
}
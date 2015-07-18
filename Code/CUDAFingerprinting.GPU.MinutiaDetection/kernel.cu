#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include "constsmacros.h"

//#define DEBUG

#ifdef DEBUG
#include "ImageLoading.cuh"
#endif

#ifndef DEBUG
extern "C"
{
	__declspec(dllexport) int GetMinutias(float* dest, int* data, double* orientation, int width, int height);
}
#endif

#ifdef DEBUG
#include <stdio.h>
#define DBGM(msg) printf("%s\n", msg)
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0);												 \
   }                                                                 \
}
#else
#define DBGM(msg) ;
#define cudaCheckError() ;
#endif

__constant__ int BLACK = 0;
__constant__ int GREY = 128;
__constant__ int WHITE = 255;

__constant__ int w;
__constant__ int h;

__device__ bool IsAvailablePixel(int x, int y)
{
	return !(x < 0 || y < 0 || x >= w || y >= h);
}

__device__ int GetPixel(int* data, int x, int y)
{
	return  !IsAvailablePixel(x, y) ?
		WHITE :
		data[(h - 1 - y) * w + x] > GREY ?
			WHITE :
			BLACK;
}

__device__ int MinutiaCode(int* data, int x, int y)
{
	if (GetPixel(data, x, y) != BLACK)
		return -1;
	//check 8-neigbourhood
	bool p[8] = {
		GetPixel(data, x, y - 1) > 0,
		GetPixel(data, x + 1, y - 1) > 0,
		GetPixel(data, x + 1, y) > 0,
		GetPixel(data, x + 1, y + 1) > 0,
		GetPixel(data, x, y + 1) > 0,
		GetPixel(data, x - 1, y + 1) > 0,
		GetPixel(data, x - 1, y) > 0,
		GetPixel(data, x - 1, y - 1) > 0,
	};

	int NeigboursCount = 0;
	for (int i = 1; i < 9; i++)
	{
		NeigboursCount += p[i % 8] ^ p[i - 1] ? 1 : 0;
	}
	NeigboursCount /= 2;
	return NeigboursCount;
}

__device__ bool InCircle(int xC, int yC, int R, int x, int y)
{
	return pow((double)(xC - x), 2) + pow((double)(yC - y), 2) < R * R;
}

__device__ double GetCorrectAngle(int* data, double* orientation, int x, int y, int NeigboursCount)
{
	double angle = orientation[(h - 1 - y) * w + x];
	float PI = 3.141592654f;
	//for 'end line' minutia
	if (NeigboursCount == 1)
	{
		if (angle > 0.0)
		{
			if ((GetPixel(data, x, y - 1) +
				GetPixel(data, x + 1, y - 1) +
				GetPixel(data, x + 1, y))
				<
				(GetPixel(data, x, y + 1) +
				GetPixel(data, x - 1, y + 1) +
				GetPixel(data, x - 1, y)))
			{
				angle += PI;
			}
		}
		else
		{
			if ((GetPixel(data, x, y + 1) +
				GetPixel(data, x + 1, y + 1) +
				GetPixel(data, x + 1, y))
				<
				(GetPixel(data, x, y - 1) +
				GetPixel(data, x - 1, y - 1) +
				GetPixel(data, x - 1, y)))
			{
				angle += PI;
			}
		}
	}
	//for 'fork' minutia
	else if (NeigboursCount == 3)
	{
		for (int r = 1; r < 16; r++)
		{
			double normal = angle + PI / 2;
			int aboveNormal = 0;
			int belowNormal = 0;

			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					if (i == j && j == 0)
					{
						continue;
					}
					if (GetPixel(data, x + j, y + i) == BLACK &&
						InCircle(x, y, r, x + j, y + i))
					{
						double deltaNormalY = - tan(normal) * j;
						if (i < deltaNormalY)
						{
							aboveNormal++;
						}
						else
						{
							belowNormal++;
						}
					}
				}
			}
			if (aboveNormal == belowNormal)
			{
				continue;//?
			}
			else
			{
				if ((aboveNormal > belowNormal &&
					tan(angle) > 0.0) ||
					(aboveNormal < belowNormal &&
					tan(angle) < 0.0))
				{
					angle += PI;
				}
				break;
			}
		}
	}
	return angle;
}

#define BLOCK_DIM 16

__global__ void ProcessPixel(float* dest, int* data, double* orientation)
{
	//x coord of image
	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	//y coord of image
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
	
	if (!IsAvailablePixel(x, y))
	{
		return;
	}

	dest[(y * w + x) * 3] = -1.0;
	dest[(y * w + x) * 3 + 1] = -1.0;
	dest[(y * w + x) * 3 + 2] = -1.0;

	int NeigboursCount = MinutiaCode(data, x, y);
	//count == 0 <=> isolated point - NOT minutia
	//count == 1 <=> 'end line' - minutia
	//count == 2 <=> part of the line - NOT minutia
	//count == 3 <=> 'fork' - minutia
	//count >= 3 <=> composit minutia - ignoring in this implementation
	bool IsMinutia = ((NeigboursCount == 1) || (NeigboursCount == 3));

	if (IsMinutia)
	{
		dest[(y * w + x) * 3] = (float)x;
		dest[(y * w + x) * 3 + 1] = (float)y;
		dest[(y * w + x) * 3 + 2] = (float)GetCorrectAngle(
			data,
			orientation,
			x,
			y,
			NeigboursCount
		);
	}
}

//shift minutias data to beginning of array
int ShrinkResult(float* dest, float* destBuffer, int width, int height)
{
	int minutiasNumber = 0;
	int size = width * height * 3;
	for (int i = 0; i < size; i += 3)
	{
		if (destBuffer[i] > -1.0)
		{
			dest[minutiasNumber * 3] = destBuffer[i];
			dest[minutiasNumber * 3 + 1] = destBuffer[i + 1];
			dest[minutiasNumber * 3 + 2] = destBuffer[i + 2];
			minutiasNumber++;
		}
	}
	free(destBuffer);
	return minutiasNumber;
}

//returns number of found minutias
//in result:
//dest[i * 3 + 0] - x coord of i's minutia
//dest[i * 3 + 1] - y coord of i's minutia
//dest[i * 3 + 2] - direction of i's minutia
int GetMinutias(float* dest, int* data, double* orientation, int width, int height)
{
	cudaMemcpyToSymbol(w, &width, sizeof(width));
	cudaCheckError();

	cudaMemcpyToSymbol(h, &height, sizeof(height));
	cudaCheckError();

	float* destBuffer = (float*)malloc(sizeof(float) * height * width * 3);

	//allocate memory on device & initialize
	float* devDest;
	cudaMalloc((void**)&devDest, sizeof(float) * height * width * 3);
	cudaCheckError();

	int* devData;
	cudaMalloc((void**)&devData, sizeof(int) * height * width);
	cudaCheckError();
	cudaMemcpy(devData, data, sizeof(int) * height * width, cudaMemcpyHostToDevice);
	cudaCheckError();

	double* devOrientation;
	cudaMalloc((void**)&devOrientation, sizeof(double) * height * width);
	cudaCheckError();
	cudaMemcpy(devOrientation, orientation, sizeof(double) * height * width, cudaMemcpyHostToDevice);
	cudaCheckError();

	int blocksRowSize = ceilMod(width, BLOCK_DIM);
	int blocksColumnSize = ceilMod(height, BLOCK_DIM);
	dim3 gridSize = dim3(blocksRowSize, blocksColumnSize);
	dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);

	ProcessPixel <<<gridSize, blockSize>>>(devDest, devData, devOrientation);

	//getting results & free device memory
	cudaMemcpy(destBuffer, devDest, sizeof(float) * height * width * 3, cudaMemcpyDeviceToHost);
	cudaCheckError();
	cudaFree(devDest);
	cudaCheckError(); 
	cudaFree(devData);
	cudaCheckError();
	cudaFree(devOrientation);
	cudaCheckError();

	return ShrinkResult(dest, destBuffer, width, height);
}

#ifdef DEBUG
int main()
{
	cudaSetDevice(0);
	int width = 0;
	int height = 0;
	int* img = loadBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\skeleton.bmp", &width, &height);//test file from folder with executable file
	
	double* orientation = (double*)malloc(sizeof(double) * width * height);

	float* minutiasArray = (float*)malloc(sizeof(float) * width * height * 3);

	int minutiasCount = GetMinutias(
		minutiasArray,
		img,
		orientation,
		width,
		height
	);

	free(img);
	free(orientation);
	free(minutiasArray);
	return 0;
}
#endif
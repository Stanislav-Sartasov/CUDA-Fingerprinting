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

struct Minutia
{
	float angle;
	int x;
	int y;
};

#ifndef DEBUG
extern "C"
{
	__declspec(dllexport) int GetMinutias(Minutia* dest, int* data, float* orientation, int width, int height);
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

#define BLOCK_DIM 16

__constant__ int BLACK = 0;
__constant__ int GREY = 128;
__constant__ int WHITE = 255;

__constant__ int w;
__constant__ int h;
/**/
__device__ inline bool IsAvailablePixel(int x, int y)
{
	return !(x < 0 || y < 0 || x >= w || y >= h);
}

__device__ inline int GetPixel(int* data, int x, int y)
{
	return  !IsAvailablePixel(x, y) ?
		WHITE :
		data[((y - blockIdx.y * BLOCK_DIM + 1) % (BLOCK_DIM + 2)) * (BLOCK_DIM + 2) +
			((x - blockIdx.x * BLOCK_DIM + 1) % (BLOCK_DIM + 2))] > GREY ?
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

__device__ inline bool InCircle(int xC, int yC, int R, int x, int y)
{
	return (xC - x) * (xC - x) + (yC - y) * (yC - y) < R * R;
}

__device__ inline int GetP(int* data, int x, int y)
{
	return  !IsAvailablePixel(x, y) ?
		WHITE :
		data[(h - 1 - y) * w + x] > GREY ?
			WHITE :
			BLACK;
}

//TODO: separate 
//first kernel: finding minutias
//second kernel: computing angle for every found minutia

__global__ void ProcessPixel(Minutia* destLines, Minutia* destForks, int* data)
{
	//x coord of image
	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	//y coord of image
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
	
	//caching
	__shared__ int Neigbourhood[(BLOCK_DIM + 2) * (BLOCK_DIM + 2)];

	if (threadIdx.y == 0 && threadIdx.x == 0)
	{
		Neigbourhood[0 * (BLOCK_DIM + 2) + 0] = GetP(data, x - 1, y - 1);
		Neigbourhood[0 * (BLOCK_DIM + 2) + 1] = GetP(data, x, y - 1);
		Neigbourhood[1 * (BLOCK_DIM + 2) + 0] = GetP(data, x - 1, y);
	}
	else if (threadIdx.y == 0 && threadIdx.x == BLOCK_DIM - 1)
	{
		Neigbourhood[0 * (BLOCK_DIM + 2) + (BLOCK_DIM + 1)] = GetP(data, x + 1, y - 1);
		Neigbourhood[0 * (BLOCK_DIM + 2) + (BLOCK_DIM)] = GetP(data, x, y - 1);
		Neigbourhood[1 * (BLOCK_DIM + 2) + (BLOCK_DIM + 1)] = GetP(data, x + 1, y);
	}
	else if (threadIdx.y == BLOCK_DIM - 1 && threadIdx.x == 0)
	{
		Neigbourhood[(BLOCK_DIM + 1) * (BLOCK_DIM + 2) + 0] = GetP(data, x - 1, y + 1);
		Neigbourhood[(BLOCK_DIM)     * (BLOCK_DIM + 2) + 0] = GetP(data, x - 1, y);
		Neigbourhood[(BLOCK_DIM + 1) * (BLOCK_DIM + 2) + 1] = GetP(data, x, y + 1);
	}
	else if (threadIdx.y == BLOCK_DIM - 1 && threadIdx.x == BLOCK_DIM - 1)
	{
		Neigbourhood[(BLOCK_DIM + 1) * (BLOCK_DIM + 2) + (BLOCK_DIM + 1)] = GetP(data, x + 1, y + 1);
		Neigbourhood[(BLOCK_DIM)     * (BLOCK_DIM + 2) + (BLOCK_DIM + 1)] = GetP(data, x + 1, y);
		Neigbourhood[(BLOCK_DIM + 1) * (BLOCK_DIM + 2) + (BLOCK_DIM)]     = GetP(data, x, y + 1);
	}
	else if (threadIdx.y == 0)
	{
		Neigbourhood[0 * (BLOCK_DIM + 2) + threadIdx.x + 1] = GetP(data, x, y - 1);
	}
	else if (threadIdx.x == 0)
	{
		Neigbourhood[(threadIdx.y + 1)* (BLOCK_DIM + 2) + 0] = GetP(data, x - 1, y);
	}
	else if (threadIdx.y == BLOCK_DIM - 1)
	{
		Neigbourhood[(BLOCK_DIM + 1) * (BLOCK_DIM + 2) + threadIdx.x + 1] = GetP(data, x, y + 1);
	}
	else if (threadIdx.x == BLOCK_DIM - 1)
	{
		Neigbourhood[(threadIdx.y + 1)* (BLOCK_DIM + 2) + (BLOCK_DIM + 1)] = GetP(data, x + 1, y);
	}

	Neigbourhood[(threadIdx.y + 1) * (BLOCK_DIM + 2) + threadIdx.x + 1] = GetP(data, x, y);
	
	__syncthreads();
	
	if (!IsAvailablePixel(x, y))
	{
		return;
	}

	int minutiasNumber = y * w + x;

	//count == 0 <=> isolated point - NOT minutia
	//count == 1 <=> 'end line' - minutia
	//count == 2 <=> part of the line - NOT minutia
	//count == 3 <=> 'fork' - minutia
	//count >= 3 <=> composit minutia - ignoring in this implementation
	int NeigboursCount = MinutiaCode(Neigbourhood, x, y);

	if (NeigboursCount == 1)
	{
		destLines[minutiasNumber].x = x;
		destLines[minutiasNumber].y = y;

		destForks[minutiasNumber].x = -1;
	}
	else if (NeigboursCount == 3)
	{
		destForks[minutiasNumber].x = x;
		destForks[minutiasNumber].y = y;

		destLines[minutiasNumber].x = -1;
	}
	else
	{
		destLines[minutiasNumber].x = -1;
		destForks[minutiasNumber].x = -1;
	}
}

__device__ inline float GetCorrectAngleForLine(int* data, float* orientation, int x, int y)
{
	float angle = orientation[(h - 1 - y) * w + x];
	float PI = 3.141592654f;
	//for 'end line' minutia
	if (angle > 0.0f)
	{
		if ((GetP(data, x, y - 1) +
			GetP(data, x + 1, y - 1) +
			GetP(data, x + 1, y))
			<
			(GetP(data, x, y + 1) +
			GetP(data, x - 1, y + 1) +
			GetP(data, x - 1, y)))
		{
			angle += PI;
		}
	}
	else
	{
		if ((GetP(data, x, y + 1) +
			GetP(data, x + 1, y + 1) +
			GetP(data, x + 1, y))
			<
			(GetP(data, x, y - 1) +
			GetP(data, x - 1, y - 1) +
			GetP(data, x - 1, y)))
		{
			angle += PI;
		}
	}
	return angle;
}

__device__ inline float GetCorrectAngleForFork(int* data, float* orientation, int x, int y)
{
	float angle = orientation[(h - 1 - y) * w + x];
	float PI = 3.141592654f;
	//for 'fork' minutia
		for (int r = 1; r < 16; r++)
		{
			float normal = angle + PI / 2;
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
					if (!InCircle(x, y, r - 1, x + j, y + i) &&
						InCircle(x, y, r, x + j, y + i) &&
						GetP(data, x + j, y + i) == BLACK)
					{
						float deltaNormalY = -tan(normal) * j;
						if (((float)i) < deltaNormalY)
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
					tan(angle) > 0.0f) ||
					(aboveNormal < belowNormal &&
					tan(angle) < 0.0f))
				{
					angle += PI;
				}
				break;
			}
		}
	return angle;
}

__global__ void ProcessAnglesForLines(Minutia* dest, int* data, float* orientation, int minutiasCount)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < minutiasCount)
	{
		dest[i].angle = GetCorrectAngleForLine(
			data,
			orientation,
			dest[i].x,
			dest[i].y
		);
	}
}

__global__ void ProcessAnglesForForks(Minutia* dest, int* data, float* orientation, int minutiasCount)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < minutiasCount)
	{
		dest[i].angle = GetCorrectAngleForFork(
			data,
			orientation,
			dest[i].x,
			dest[i].y
		);
	}
}

//shift minutias data to beginning of array
int ShrinkResult(Minutia* dest, Minutia* destBuffer, int width, int height)
{
	int minutiasNumber = 0;
	int size = width * height;
	for (int i = 0; i < size; i++)
	{
		if (destBuffer[i].x > -1)
		{
			dest[minutiasNumber].x = destBuffer[i].x;
			dest[minutiasNumber].y = destBuffer[i].y;
			dest[minutiasNumber].angle = destBuffer[i].angle;
			minutiasNumber++;
		}
	}
	return minutiasNumber;
}

//returns number of found minutias
//in result:
//dest[i * 3 + 0] - x coord of i's minutia
//dest[i * 3 + 1] - y coord of i's minutia
//dest[i * 3 + 2] - direction of i's minutia
int GetMinutias(Minutia* dest, int* data, float* orientation, int width, int height)
{
	cudaMemcpyToSymbol(w, &width, sizeof(width));
	cudaCheckError();

	cudaMemcpyToSymbol(h, &height, sizeof(height));
	cudaCheckError();

	Minutia* destBufferLines = (Minutia*)malloc(sizeof(Minutia) * height * width);
	Minutia* destBufferForks = (Minutia*)malloc(sizeof(Minutia) * height * width);

	//allocate memory on device & initialize
	Minutia* devDestLines;
	cudaMalloc((void**)&devDestLines, sizeof(Minutia) * height * width);
	cudaCheckError();

	Minutia* devDestForks;
	cudaMalloc((void**)&devDestForks, sizeof(Minutia) * height * width);
	cudaCheckError();

	int* devData;
	cudaMalloc((void**)&devData, sizeof(int) * height * width);
	cudaCheckError();
	cudaMemcpy(devData, data, sizeof(int) * height * width, cudaMemcpyHostToDevice);
	cudaCheckError();

	float* devOrientation;
	cudaMalloc((void**)&devOrientation, sizeof(float) * height * width);
	cudaCheckError();
	cudaMemcpy(devOrientation, orientation, sizeof(float) * height * width, cudaMemcpyHostToDevice);
	cudaCheckError();

	int blocksRowSize = ceilMod(width, BLOCK_DIM);
	int blocksColumnSize = ceilMod(height, BLOCK_DIM);
	dim3 gridSize = dim3(blocksRowSize, blocksColumnSize);
	dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);

	ProcessPixel << <gridSize, blockSize >> >(devDestLines, devDestForks, devData);

	//getting results & free device memory
	cudaMemcpy(destBufferLines, devDestLines, sizeof(Minutia) * height * width, cudaMemcpyDeviceToHost);
	cudaCheckError();
	cudaMemcpy(destBufferForks, devDestForks, sizeof(Minutia) * height * width, cudaMemcpyDeviceToHost);
	cudaCheckError();

	Minutia* destBufferLinesShrinked = (Minutia*)malloc(sizeof(Minutia) * height * width);
	Minutia* destBufferForksShrinked = (Minutia*)malloc(sizeof(Minutia) * height * width);

	int minutiasCountLines = ShrinkResult(destBufferLinesShrinked, destBufferLines, width, height);
	int minutiasCountForks = ShrinkResult(destBufferForksShrinked, destBufferForks, width, height);

	free(destBufferLines);
	free(destBufferForks);

	cudaMemcpy(devDestLines, destBufferLinesShrinked, sizeof(Minutia) * height * width, cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(devDestForks, destBufferForksShrinked, sizeof(Minutia) * height * width, cudaMemcpyHostToDevice);
	cudaCheckError();

	ProcessAnglesForLines << <dim3(ceilMod(minutiasCountLines, BLOCK_DIM)), dim3(BLOCK_DIM, 1, 1) >> >(devDestLines, devData, devOrientation, minutiasCountLines);
	ProcessAnglesForForks << <dim3(ceilMod(minutiasCountForks, BLOCK_DIM)), dim3(BLOCK_DIM, 1, 1) >> >(devDestForks, devData, devOrientation, minutiasCountForks);

	cudaMemcpy(dest, devDestLines, sizeof(Minutia) * minutiasCountLines, cudaMemcpyDeviceToHost);
	cudaCheckError();
	cudaMemcpy(dest + minutiasCountLines, devDestForks, sizeof(Minutia) * minutiasCountForks, cudaMemcpyDeviceToHost);
	cudaCheckError();
	cudaFree(devDestLines);
	cudaCheckError();
	cudaFree(devDestForks);
	cudaCheckError();
	cudaFree(devData);
	cudaCheckError();
	cudaFree(devOrientation);
	cudaCheckError();

	return minutiasCountLines + minutiasCountForks;
}

#ifdef DEBUG
void prntArr(Minutia arr[], int size)
{
	while (-1 < --size)
	{
		printf("X=%3.d, Y=%3.d, ANGLE=%e     ", arr[size].x, arr[size].y, arr[size].angle);
		if (size % 2 == 0)
		{
			printf("\n");
		}
	}
}

void initOr(float* or, int size)
{
	while (-1 < --size)
	{
		or[size] = (float)size;
	}
}

int* overlapMinutias(int* img, Minutia ms[], int minSize, int width, int height)
{
	while (-1 < --minSize)
	{
		img[(height - 1 - ms[minSize].y) * width + ms[minSize].x] = 128;
	}
	return img;
}

int main()
{
	cudaSetDevice(0);
	int width = 0;
	int height = 0;
	int* img = loadBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\skeleton.bmp", &width, &height);//test file from folder with executable file
	
	float* orientation = (float*)malloc(sizeof(float) * width * height);
	initOr(orientation, width * height);
	
	Minutia* minutiasArray = (Minutia*)malloc(sizeof(Minutia) * width * height);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	int minutiasCount = GetMinutias(
		minutiasArray,
		img,
		orientation,
		width,
		height
	);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);

	printf("\nTotal GetMinutias execution time:                %e ms\nBLOCK DIM: %d\nMinutias found: %d\n", time, BLOCK_DIM, minutiasCount);
	
	//prntArr(minutiasArray, minutiasCount);

	overlapMinutias(img, minutiasArray, minutiasCount, width, height);
	saveBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\resultCUDA.bmp", img, width, height);
	system("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\resultCUDA.bmp");

	free(img);
	free(orientation);
	free(minutiasArray);
	return 0;
}
#endif
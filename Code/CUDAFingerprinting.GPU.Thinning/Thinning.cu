#include "Thinning.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "ThinnerUtils.h"
#include "constsmacros.h"

//#define DEBUG

#ifdef DEBUG
#include "ImageLoading.cuh"
#include "Utils.h"
#define DBGM(msg) printf("%s\n", msg)
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
  }                                                                 \
}
#else
#define DBGM(msg) ;
#define cudaCheckError() ;
#endif

//patterns
__constant__ PixelType constP[14 * 3 * 3];

#define PTTRN(p, i, y, x) p[i * 9 + y * 3 + x]

__device__ PixelType PATTERN(int i, int x, int y)
{
	return constP[i * 9 + y * 3 + x];
}

__host__ void InitPatterns()
{
	int patternsArraySize = sizeof(PixelType) * 14 * 3 * 3;

	PixelType* p = (PixelType*)malloc(patternsArraySize);
	//a
	PTTRN(p, 0, 0, 0) = PixelType::FILLED; PTTRN(p, 0, 0, 1) = PixelType::FILLED; PTTRN(p, 0, 0, 2) = PixelType::AT_LEAST_ONE_EMPTY;
	PTTRN(p, 0, 1, 0) = PixelType::FILLED; PTTRN(p, 0, 1, 1) = PixelType::CENTER; PTTRN(p, 0, 1, 2) = PixelType::EMPTY;
	PTTRN(p, 0, 2, 0) = PixelType::FILLED; PTTRN(p, 0, 2, 1) = PixelType::FILLED; PTTRN(p, 0, 2, 2) = PixelType::AT_LEAST_ONE_EMPTY;
	//b
	PTTRN(p, 1, 0, 0) = PixelType::FILLED;			   PTTRN(p, 1, 0, 1) = PixelType::FILLED; PTTRN(p, 1, 0, 2) = PixelType::FILLED;
	PTTRN(p, 1, 1, 0) = PixelType::FILLED;			   PTTRN(p, 1, 1, 1) = PixelType::CENTER; PTTRN(p, 1, 1, 2) = PixelType::FILLED;
	PTTRN(p, 1, 2, 0) = PixelType::AT_LEAST_ONE_EMPTY; PTTRN(p, 1, 2, 1) = PixelType::EMPTY;  PTTRN(p, 1, 2, 2) = PixelType::AT_LEAST_ONE_EMPTY;
	//c - needs special processing
	PTTRN(p, 2, 0, 0) = PixelType::AT_LEAST_ONE_EMPTY; PTTRN(p, 2, 0, 1) = PixelType::FILLED; PTTRN(p, 2, 0, 2) = PixelType::FILLED;
	PTTRN(p, 2, 1, 0) = PixelType::EMPTY;			   PTTRN(p, 2, 1, 1) = PixelType::CENTER; PTTRN(p, 2, 1, 2) = PixelType::FILLED;//PixelType.FILLED 
	PTTRN(p, 2, 2, 0) = PixelType::AT_LEAST_ONE_EMPTY; PTTRN(p, 2, 2, 1) = PixelType::FILLED; PTTRN(p, 2, 2, 2) = PixelType::FILLED;
	//d - needs special processing
	PTTRN(p, 3, 0, 0) = PixelType::AT_LEAST_ONE_EMPTY; PTTRN(p, 3, 0, 1) = PixelType::EMPTY;  PTTRN(p, 3, 0, 2) = PixelType::AT_LEAST_ONE_EMPTY;
	PTTRN(p, 3, 1, 0) = PixelType::FILLED;			   PTTRN(p, 3, 1, 1) = PixelType::CENTER; PTTRN(p, 3, 1, 2) = PixelType::FILLED;
	PTTRN(p, 3, 2, 0) = PixelType::FILLED;			   PTTRN(p, 3, 2, 1) = PixelType::FILLED; PTTRN(p, 3, 2, 2) = PixelType::FILLED;
													    	//PixelType.FILLED
	//e
	PTTRN(p, 4, 0, 0) = PixelType::ANY;    PTTRN(p, 4, 0, 1) = PixelType::EMPTY;  PTTRN(p, 4, 0, 2) = PixelType::EMPTY;
	PTTRN(p, 4, 1, 0) = PixelType::FILLED; PTTRN(p, 4, 1, 1) = PixelType::CENTER; PTTRN(p, 4, 1, 2) = PixelType::EMPTY;
	PTTRN(p, 4, 2, 0) = PixelType::ANY;    PTTRN(p, 4, 2, 1) = PixelType::FILLED; PTTRN(p, 4, 2, 2) = PixelType::ANY;
	//f
	PTTRN(p, 5, 0, 0) = PixelType::ANY;   PTTRN(p, 5, 0, 1) = PixelType::FILLED; PTTRN(p, 5, 0, 2) = PixelType::FILLED;
	PTTRN(p, 5, 1, 0) = PixelType::EMPTY; PTTRN(p, 5, 1, 1) = PixelType::CENTER; PTTRN(p, 5, 1, 2) = PixelType::FILLED;
	PTTRN(p, 5, 2, 0) = PixelType::EMPTY; PTTRN(p, 5, 2, 1) = PixelType::EMPTY;  PTTRN(p, 5, 2, 2) = PixelType::ANY;
	//g
	PTTRN(p, 6, 0, 0) = PixelType::EMPTY; PTTRN(p, 6, 0, 1) = PixelType::FILLED; PTTRN(p, 6, 0, 2) = PixelType::EMPTY;
	PTTRN(p, 6, 1, 0) = PixelType::EMPTY; PTTRN(p, 6, 1, 1) = PixelType::CENTER; PTTRN(p, 6, 1, 2) = PixelType::FILLED;
	PTTRN(p, 6, 2, 0) = PixelType::EMPTY; PTTRN(p, 6, 2, 1) = PixelType::EMPTY;  PTTRN(p, 6, 2, 2) = PixelType::EMPTY;
	//h
	PTTRN(p, 7, 0, 0) = PixelType::ANY;    PTTRN(p, 7, 0, 1) = PixelType::FILLED; PTTRN(p, 7, 0, 2) = PixelType::ANY;
	PTTRN(p, 7, 1, 0) = PixelType::FILLED; PTTRN(p, 7, 1, 1) = PixelType::CENTER; PTTRN(p, 7, 1, 2) = PixelType::EMPTY;
	PTTRN(p, 7, 2, 0) = PixelType::ANY;    PTTRN(p, 7, 2, 1) = PixelType::EMPTY;  PTTRN(p, 7, 2, 2) = PixelType::EMPTY;
	//i
	PTTRN(p, 8, 0, 0) = PixelType::EMPTY; PTTRN(p, 8, 0, 1) = PixelType::EMPTY;  PTTRN(p, 8, 0, 2) = PixelType::ANY;
	PTTRN(p, 8, 1, 0) = PixelType::EMPTY; PTTRN(p, 8, 1, 1) = PixelType::CENTER; PTTRN(p, 8, 1, 2) = PixelType::FILLED;
	PTTRN(p, 8, 2, 0) = PixelType::ANY;   PTTRN(p, 8, 2, 1) = PixelType::FILLED; PTTRN(p, 8, 2, 2) = PixelType::FILLED;
	//j
	PTTRN(p, 9, 0, 0) = PixelType::EMPTY; PTTRN(p, 9, 0, 1) = PixelType::EMPTY;  PTTRN(p, 9, 0, 2) = PixelType::EMPTY;
	PTTRN(p, 9, 1, 0) = PixelType::EMPTY; PTTRN(p, 9, 1, 1) = PixelType::CENTER; PTTRN(p, 9, 1, 2) = PixelType::FILLED;
	PTTRN(p, 9, 2, 0) = PixelType::EMPTY; PTTRN(p, 9, 2, 1) = PixelType::FILLED; PTTRN(p, 9, 2, 2) = PixelType::EMPTY;
	//k
	PTTRN(p, 10, 0, 0) = PixelType::EMPTY;  PTTRN(p, 10, 0, 1) = PixelType::EMPTY;  PTTRN(p, 10, 0, 2) = PixelType::EMPTY;
	PTTRN(p, 10, 1, 0) = PixelType::EMPTY;  PTTRN(p, 10, 1, 1) = PixelType::CENTER; PTTRN(p, 10, 1, 2) = PixelType::EMPTY;
	PTTRN(p, 10, 2, 0) = PixelType::FILLED; PTTRN(p, 10, 2, 1) = PixelType::FILLED; PTTRN(p, 10, 2, 2) = PixelType::FILLED;
	//l
	PTTRN(p, 11, 0, 0) = PixelType::FILLED; PTTRN(p, 11, 0, 1) = PixelType::EMPTY;  PTTRN(p, 11, 0, 2) = PixelType::EMPTY;
	PTTRN(p, 11, 1, 0) = PixelType::FILLED; PTTRN(p, 11, 1, 1) = PixelType::CENTER; PTTRN(p, 11, 1, 2) = PixelType::EMPTY;
	PTTRN(p, 11, 2, 0) = PixelType::FILLED; PTTRN(p, 11, 2, 1) = PixelType::EMPTY;  PTTRN(p, 11, 2, 2) = PixelType::EMPTY;
	//m
	PTTRN(p, 12, 0, 0) = PixelType::FILLED; PTTRN(p, 12, 0, 1) = PixelType::FILLED; PTTRN(p, 12, 0, 2) = PixelType::FILLED;
	PTTRN(p, 12, 1, 0) = PixelType::EMPTY;  PTTRN(p, 12, 1, 1) = PixelType::CENTER; PTTRN(p, 12, 1, 2) = PixelType::EMPTY;
	PTTRN(p, 12, 2, 0) = PixelType::EMPTY;  PTTRN(p, 12, 2, 1) = PixelType::EMPTY;  PTTRN(p, 12, 2, 2) = PixelType::EMPTY;
	//n
	PTTRN(p, 13, 0, 0) = PixelType::EMPTY; PTTRN(p, 13, 0, 1) = PixelType::EMPTY;  PTTRN(p, 13, 0, 2) = PixelType::FILLED;
	PTTRN(p, 13, 1, 0) = PixelType::EMPTY; PTTRN(p, 13, 1, 1) = PixelType::CENTER; PTTRN(p, 13, 1, 2) = PixelType::FILLED;
	PTTRN(p, 13, 2, 0) = PixelType::EMPTY; PTTRN(p, 13, 2, 1) = PixelType::EMPTY;  PTTRN(p, 13, 2, 2) = PixelType::FILLED;
	
	cudaMemcpyToSymbol(constP, p, patternsArraySize);
	cudaCheckError();

	free(p);
}

__device__ double GetPixel(double* array, int x, int y, int width, int height)
{
	return (x < 0 || y < 0 || x >= width || y >= height) ?
		WHITE :
		array[y * width + x] > 128.0 ?
			WHITE :
			BLACK;
}

__device__ void SetPixel(double* array, int x, int y, int width, int height, double value)
{
	if (x < 0 || y < 0 || x >= width || y >= height) return;
	array[y * width + x] = value;
}

__device__ bool AreEqual(double value, PixelType patternPixel)
{
	switch (patternPixel)
	{
	case PixelType::FILLED:
	{
		if (value == BLACK)
			return true;
		break;
	}
	case PixelType::EMPTY:
	{
		if (value == WHITE)
			return true;
		break;
	}
	case PixelType::AT_LEAST_ONE_EMPTY://y
		return true;
	case PixelType::CENTER://c
		if (value == BLACK)
			return true;
		break;
	case PixelType::ANY://x
		return true;
	default:
		break;
	}
	return false;
}

//-1 - no match
__device__ int MatchPattern(double* array, int x, int y, int width, int height)
{
	if (GetPixel(array, x, y, width, height) == WHITE) return -1;
	for (int i = 0; i < 14; i++)
	{
		bool yInPattern = false;
		int yWhiteCounter = 0;
		bool bad = false;
		for (int dY = -1; dY < 2; dY++)
		{
			if (bad)
			{
				break;
			}
			for (int dX = -1; dX < 2; dX++)
			{
				if (PATTERN(i, 1 + dX, 1 + dY) == PixelType::AT_LEAST_ONE_EMPTY)
				{
					yInPattern = true;
					yWhiteCounter += GetPixel(array, x + dX, y + dY, width, height) == WHITE ? 1 : 0;
					continue;
				}
				if (!AreEqual(GetPixel(array, x + dX, y + dY, width, height), PATTERN(i, 1 + dX, 1 + dY)))
				{
					bad = true;
					break;
				}
			}
		}
		if (bad)
		{
			continue;
		}
		if (yInPattern && yWhiteCounter == 0)
		{
			continue;
		}
		if (i == 2 && !AreEqual(GetPixel(array, x + 2, y, width, height), PixelType::FILLED))
		{
			continue;
		}
		else if (i == 3 && !AreEqual(GetPixel(array, x, y + 2, width, height), PixelType::FILLED))
		{
			continue;
		}
		return i;
	}
	return -1;
}

#define BLOCK_DIM 8

__global__ void onePass(double* array, double* buffer, int width, int height, bool* wereNotChangesInBlock)
{
	__shared__ bool wereNotChanges[BLOCK_DIM][BLOCK_DIM];

	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;

	wereNotChanges[threadIdx.y][threadIdx.x] = true;

	int pattern = MatchPattern(array, x, y, width, height);
	if (pattern != -1)
	{
		SetPixel(buffer, x, y, width, height, WHITE);
		wereNotChanges[threadIdx.y][threadIdx.x] = false;
	}
	__syncthreads();
	bool fl = true;
	
	for (int i = 0; i < BLOCK_DIM; i++)
	{
		for (int j = 0; j < BLOCK_DIM; j++)
		{
			if (!wereNotChanges[i][j])
			{
				fl = false;
				break;
			}
		}
		if (!fl)
		{
			break;
		}
	}
	wereNotChangesInBlock[blockIdx.y * gridDim.x + blockIdx.x] = fl;
}

__device__ bool MatchPattern(double* array, int x, int y, int width, int height, int i)
{
	if (GetPixel(array, x, y, width, height) != WHITE)
	{
		bool yInPattern = false;
		int yWhiteCounter = 0;
		bool bad = false;
		for (int dY = -1; dY < 2; dY++)
		{
			if (bad)
			{
				break;
			}
			for (int dX = -1; dX < 2; dX++)
			{
				if (PATTERN(i, 1 + dX, 1 + dY) == PixelType::AT_LEAST_ONE_EMPTY)
				{
					yInPattern = true;
					yWhiteCounter += GetPixel(array, x + dX, y + dY, width, height) == WHITE ? 1 : 0;
					continue;
				}
				if (!AreEqual(GetPixel(array, x + dX, y + dY, width, height), PATTERN(i, 1 + dX, 1 + dY)))
				{
					bad = true;
					break;
				}
			}
		}
		if (!(bad ||
			(yInPattern && yWhiteCounter == 0) ||
			(i == 2 && !AreEqual(GetPixel(array, x + 2, y, width, height), PixelType::FILLED)) ||
			(i == 3 && !AreEqual(GetPixel(array, x, y + 2, width, height), PixelType::FILLED))))
		{
			return true;
		}
		else
		{
			return false;
		}
	}
	else
	{
		return false;
	}
}

__global__ void newOnePass(double* array, double* buffer, int width, int height, bool* wereNotChangesInBlock)
{
	__shared__ bool wereNotChanges[BLOCK_DIM][BLOCK_DIM];
	__shared__ bool isPatternI[BLOCK_DIM][BLOCK_DIM][14];

	//x coord of image
	int x = blockIdx.x * BLOCK_DIM + threadIdx.x;
	//y coord of image
	int y = blockIdx.y * BLOCK_DIM + threadIdx.y;
	//pattern id
	int pId = threadIdx.z;

	isPatternI[threadIdx.y][threadIdx.x][threadIdx.z] = MatchPattern(array, x, y, width, height, pId);
	__syncthreads();
	bool isPattern = false;
	for (int i = 0; i < 14; i++)
	{
		if (isPatternI[threadIdx.y][threadIdx.x][i])
		{
			isPattern = true;
			break;
		}
	}
	__syncthreads();

	wereNotChanges[threadIdx.y][threadIdx.x] = true;

	if (isPattern)
	{
		SetPixel(buffer, x, y, width, height, WHITE);
		wereNotChanges[threadIdx.y][threadIdx.x] = false;
	}
	__syncthreads();
	bool fl = true;

	for (int i = 0; i < BLOCK_DIM; i++)
	{
		for (int j = 0; j < BLOCK_DIM; j++)
		{
			if (!wereNotChanges[i][j])
			{
				fl = false;
				break;
			}
		}
		if (!fl)
		{
			break;
		}
	}
	wereNotChangesInBlock[blockIdx.y * gridDim.x + blockIdx.x] = fl;
}

__host__ double** Thin(double** arr, int width, int height)
{
#ifdef DEBUG
	int stepC = 0;
#endif
	double* arr1D = copy2DArrayTo1D(arr, width, height);	
	InitPatterns();
	bool isSkeleton;
	double* buffer = copy1DArray(arr1D, width * height);
	free(arr1D);
	do
	{
		int blocksRowSize = ceilMod(width, BLOCK_DIM);
		int blocksColumnSize = ceilMod(height, BLOCK_DIM);
		//allocate memory on host
		bool* wereNotChangesInBlock = (bool*)malloc(
				sizeof(bool) * 
				blocksColumnSize *
				blocksRowSize);
		
		double* devA;
		double* devBuffer;
		bool* devWereNotChangesInBlock;

		//allocate memory on device & initialize
		cudaMalloc((void**)&devA, sizeof(double) * height * width);
		cudaCheckError();
		cudaMemcpy(devA, buffer, sizeof(double) * height * width, cudaMemcpyHostToDevice);
		cudaCheckError();

		cudaMalloc((void**)&devBuffer, sizeof(double) * height * width);
		cudaCheckError();
		cudaMemcpy(devBuffer, buffer, sizeof(double) * height * width, cudaMemcpyHostToDevice);
		cudaCheckError();

		cudaMalloc((void**)&devWereNotChangesInBlock, 
				sizeof(bool) * 
				blocksColumnSize *
				blocksRowSize);
		cudaCheckError();
		isSkeleton = true;
		dim3 gridSize = dim3(blocksRowSize, blocksColumnSize);
		dim3 blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 14);
		//call kernel function
		//with parallel processing of patterns
		newOnePass <<<gridSize, blockSize>>>(devA, devBuffer, width, height, devWereNotChangesInBlock);
		//without parallel processing of patterns
		//blockSize = dim3(BLOCK_DIM, BLOCK_DIM, 1);
		//onePass << <gridSize, blockSize >> >(devA, devBuffer, width, height, devWereNotChangesInBlock);

		//getting results & free device memory
		cudaMemcpy(buffer, devBuffer, sizeof(double) * height * width, cudaMemcpyDeviceToHost);
		cudaCheckError();
		cudaFree(devBuffer);
		cudaCheckError();
		cudaFree(devA);
		cudaCheckError();
		cudaMemcpy(wereNotChangesInBlock, 
				devWereNotChangesInBlock, 
				sizeof(bool) * 
				blocksColumnSize *
				blocksRowSize,
				cudaMemcpyDeviceToHost);
		cudaCheckError();
		cudaFree(devWereNotChangesInBlock);
		cudaCheckError();
		//processing flags of changes
		for (int i = 0; i < blocksColumnSize * blocksRowSize; i++)
		{
			if (!wereNotChangesInBlock[i])
			{
				isSkeleton = false;
				break;
			}
		}

		free(wereNotChangesInBlock);
#ifdef DEBUG
		if (!isSkeleton)
		{
			//thinning image trace
			int widthDBG = width;
			int heightDBG = height;
			int* img = doubleToIntArray(arr, width, height);
			//'trace' folder must exist
			char path[] = "D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\trace\\step00.bmp\0";
			path[57] = '0' + stepC % 10;
			path[56] = '0' + stepC / 10;
			stepC++;
			int* res = OverlapArrays(
				doubleToIntArray(buffer, width, height),
				img, 
				widthDBG, heightDBG
			);
			saveBmp(path, res, widthDBG, heightDBG);
			free(img);
			free(res);
		}
#endif
	} while (!isSkeleton);

	return copy1DArrayTo2D(buffer, width, height);
}
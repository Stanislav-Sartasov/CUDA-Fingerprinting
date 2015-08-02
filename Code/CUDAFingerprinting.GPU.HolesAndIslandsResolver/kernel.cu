#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define DEBUG

#ifdef DEBUG
#include "ImageLoading.cuh"
#endif

__constant__ int BLACK = 0;
__constant__ int GREY = 128;
__constant__ int WHITE = 255;

int hBLACK = 0;
int hGREY = 128;
int hWHITE = 255;

__device__ int GetPixel(int* data, int x, int y, int width, int height)
{
	return (x < 0 || y < 0 || x >= width || y >= height) ?
		-1 :
		data[y * width + x];
}

__device__ int GetArea(int* area, int x, int y, int width, int height)
{
	return GetPixel(area, x, y, width, height);
}

__host__ int hGetPixel(int* data, int x, int y, int width, int height)
{
	return (x < 0 || y < 0 || x >= width || y >= height) ?
		-1 :
		data[y * width + x];
}

__host__ int hGetArea(int* area, int x, int y, int width, int height)
{
	return hGetPixel(area, x, y, width, height);
}

__device__ int NumberOfAreas;

__device__ int GetAreaRoot(int area, int* Allies)
{
	if (Allies[area] == area)
	{
		return area;
	}
	else
	{
		return GetAreaRoot(Allies[area], Allies);
	}
}

__device__ int GetAreaSize(int area, int* AreasSize, int* Allies)
{
	int sum = 0;
	int root = GetAreaRoot(area, Allies);
	for (int i = 0; i < NumberOfAreas; i++)
	{
		if (GetAreaRoot(i, Allies) == root)
		{
			sum += AreasSize[i];
		}
	}
	return sum;
}

__global__ void Preprocessing(int* data, 
	int* w, int* h,
	int* Areas,
	int* AreasSize,
	int* Allies,
	int* outNumberOfAreas)
{
	int width = *w;
	int height = *h;

	NumberOfAreas = 0;

	Allies[0] = 0;
	AreasSize[0] = 0;
	Areas[0] = 0;

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int leftPixel = GetPixel(data, x - 1, y, width, height);
			int topPixel = GetPixel(data, x, y - 1, width, height);
			int currentPixel = GetPixel(data, x, y, width, height);

			if (leftPixel == topPixel)
			{
				if (topPixel != currentPixel)
				{
					Allies[NumberOfAreas] = NumberOfAreas;
					AreasSize[NumberOfAreas] = 1;
					Areas[y * width + x] = NumberOfAreas;
					NumberOfAreas++;
				}
				else
				{
					int leftArea = GetArea(Areas, x - 1, y, width, height);
					int topArea = GetArea(Areas, x, y - 1, width, height);

					if (topArea != leftArea)
					{
						int tAR = GetAreaRoot(topArea, Allies);
						int lAR = GetAreaRoot(leftArea, Allies);
						Allies[lAR < tAR ? tAR : lAR] = lAR < tAR ? lAR : tAR;
					}

					AreasSize[leftArea]++;
					Areas[y * width + x] = leftArea;
				}
			}
			else
			{
				if (topPixel == currentPixel)
				{
					int topArea = GetArea(Areas, x, y - 1, width, height);
					AreasSize[topArea]++;
					Areas[y * width + x] = topArea;
				}
				else
				{
					int leftArea = GetArea(Areas, x - 1, y, width, height);
					if (leftArea != -1)
					{
						AreasSize[leftArea]++;
						Areas[y * width + x] = leftArea;
					}
					else
					{
						Allies[NumberOfAreas] = NumberOfAreas;
						AreasSize[NumberOfAreas] = 1;
						Areas[y * width + x] = NumberOfAreas;
						NumberOfAreas++;
					}
				}
			}
		}
	}
	*outNumberOfAreas = NumberOfAreas;
}

int hGetAreaRoot(int area, int* Allies)
{
	if (Allies[area] == area)
	{
		return area;
	}
	else
	{
		return hGetAreaRoot(Allies[area], Allies);
	}
}

int hGetAreaSize(int area, int* AreasSize, int* Allies, int hNumberOfAreas)
{
	int sum = 0;
	int root = hGetAreaRoot(area, Allies);
	for (int i = 0; i < hNumberOfAreas; i++)
	{
		if (hGetAreaRoot(i, Allies) == root)
		{
			sum += AreasSize[i];
		}
	}
	return sum;
}


#ifdef DEBUG

void Binarization(int BARRIER, int* data, int size)
{
	for (int i = 0; i < size; i++)
	{
		data[i] = data[i] <= BARRIER ? 0 : 255;
	}
}

void WriteArray(int* data, int width, int height)
{
	for (int i = 0; i < width * height; i++)
	{
		if (i % width == 0)
			printf("\n");
		printf("%c", data[i] == 0 ? '*' : '=');
	}
}

int main()
{
	cudaSetDevice(0);
	int width = 0;
	int height = 0;
	int* img = loadBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\CUDAFingerprinting.ImageEnhancement.Tests\\Resources\\f.bmp", &width, &height);//test file from folder with executable file
	//WriteArray(img, width, height);
	Binarization(128, img, width * height);

	saveBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\sourceCUDA.bmp", img, width, height);
	system("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\sourceCUDA.bmp");

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//
	int threshold = 16;

	int* devAreas;
	cudaMalloc((void**)&devAreas, sizeof(int) * height * width);
	int* devAreasSize;
	cudaMalloc((void**)&devAreasSize, sizeof(int) * height * width);
	int* devAllies;
	cudaMalloc((void**)&devAllies, sizeof(int) * height * width);

	int* devImg;
	cudaMalloc((void**)&devImg, sizeof(int) * height * width);
	cudaMemcpy(devImg, img, sizeof(int) * height * width, cudaMemcpyHostToDevice);

	int* devWidth;
	cudaMalloc((void**)&devWidth, sizeof(int));
	cudaMemcpy(devWidth, &width, sizeof(int), cudaMemcpyHostToDevice);
	int* devHeight;
	cudaMalloc((void**)&devHeight, sizeof(int));
	cudaMemcpy(devHeight, &height, sizeof(int), cudaMemcpyHostToDevice);

	int* devNumberOfAreas;
	cudaMalloc((void**)&devNumberOfAreas, sizeof(int));

	Preprocessing << <1, 1 >> >(devImg, devWidth, devHeight, devAreas, devAreasSize, devAllies, devNumberOfAreas);

	cudaFree(devImg);
	cudaFree(devWidth);
	cudaFree(devHeight);

	int* Areas = (int*)malloc(sizeof(int) * width * height);
	int* AreasSize = (int*)malloc(sizeof(int) * width * height);
	int* Allies = (int*)malloc(sizeof(int) * width * height);
	cudaMemcpy(Areas, devAreas, sizeof(int) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(AreasSize, devAreasSize, sizeof(int) * height * width, cudaMemcpyDeviceToHost);
	cudaMemcpy(Allies, devAllies, sizeof(int) * height * width, cudaMemcpyDeviceToHost);

	int hNumberOfAreas;
	cudaMemcpy(&hNumberOfAreas, devNumberOfAreas, sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(devAreas);
	cudaFree(devAreasSize);
	cudaFree(devAllies);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (hGetPixel(img, x, y, width, height) == hWHITE &&
				hGetAreaSize(
					hGetArea(Areas, x, y, width, height),
					AreasSize,
					Allies,
					hNumberOfAreas
				) < threshold)
			{
				img[y * width + x] = hBLACK;
			}
		}
	}
	//
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);

	//WriteArray(img, width, height);

	printf("\nTotal GetMinutias execution time:                %e ms\n", time);

	//prntArr(minutiasArray, minutiasCount);

	saveBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\resultCUDA.bmp", img, width, height);
	system("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\resultCUDA.bmp");

	return 0;
}
#endif
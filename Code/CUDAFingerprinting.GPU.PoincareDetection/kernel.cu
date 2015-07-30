
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "constsmacros.h"
#include "CUDAArray.cuh"
#include "OrientationField.cuh"
#include "ImageLoading.cuh"
#include <math.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#define PI 3.141592653589

using namespace std;

extern "C"
{
	__declspec(dllexport)  void PoincareDetect(int* src, int width, int height, int blockSize,  float* orientated, int* dist);
}


__device__  void  AngleSum(int x, int y, int width, int height, int fieldSize, int orientationArrayLength, CUDAArray<int> img, CUDAArray<float> oriented, CUDAArray<float> orient16)
{
	
	float value;
	float OrrArr[16];
	float angleSum;
	
	for (short i = 0; i < orientationArrayLength / 4; i++)
	{
		int left = x - fieldSize / 2;
		int right = y + i - fieldSize / 2;
		//if (x - fieldSize / 2 >= 0 && y + i - fieldSize / 2 >= 0 && y + i - fieldSize / 2 < height && x - fieldSize / 2 < width)
		//{
		float value = oriented.At(50, 50/*right*/);
			OrrArr[i] = value;
		//}
		//else
		//{
		//	OrrArr[i] = 0.0f;
		//}
	}
//
//	for (short i = orientationArrayLength / 4; i < orientationArrayLength / 2; i++)
//	{
//		int left = x - fieldSize / 2 + i - orientationArrayLength / 4;
//		int right = y + fieldSize / 2;
//		if (left >= 0 && right >= 0 && right < height && left < width)
//		{
//			OrrArr[i] = oriented.At(left, right);
//		}
//		else
//		{
//			OrrArr[i] = 0.0f;
//		}
//	}
//
//	for (short i = orientationArrayLength / 2; i < orientationArrayLength / 2 + orientationArrayLength / 4; i++)
//	{
//		int left = x + fieldSize / 2;
//		int right = y - fieldSize / 2 - (i - 3 * orientationArrayLength / 4);
//		if (left >= 0 && right >= 0 && right < height && left < width)
//		{
//			OrrArr[i] = oriented.At(left, right);
//		}
//		else
//		{
//			OrrArr[i] = 0.0f;
//		}
//	}
//
//	for (short i = orientationArrayLength / 4 + orientationArrayLength / 2; i < orientationArrayLength; i++)
//	{
//		int left = x - fieldSize / 2 - (i - orientationArrayLength);
//		int right = y - fieldSize / 2;
//		if (left >= 0 && right >= 0 && right < height && left < width)
//		{
//			OrrArr[i] = oriented.At(left, right);
//		}
//		else
//		{
//			OrrArr[i] = 0.0f;
//		}
//	}
////	__syncthreads();
	/*angleSum = 0.0f;
	for (short i = 0; i < 16; i++)
	{

		if (abs(-OrrArr[i] + OrrArr[(i + 1) % orientationArrayLength]) <
			abs((float)PI + (-OrrArr[i] + OrrArr[(i + 1) % orientationArrayLength])))
		{
			angleSum += -OrrArr[i] + OrrArr[(i + 1) % orientationArrayLength];
			//angleSum += 10.0f;
		}
		else
		{
			angleSum += (float)PI + (-OrrArr[i] + OrrArr[(i + 1) % orientationArrayLength]);
		}
	}
	//__syncthreads();
	return angleSum;*/
	for (int i = 0; i < 16; i++)
	{
		float value2 = oriented.At(50, 50);//OrrArr[i];
		orient16.SetAt(i, 0, value2);
	}
}

__global__ void Detect(int x, int y, CUDAArray<int> img, int blockSize, CUDAArray<float> oriented, int orientationArrayLength, CUDAArray<float> orient16)
{
	//int column = defaultColumn();
	//int row = defaultRow();
	//float angleSum;
//	angleSum = 10.0f;
	//int row = 52;
	//int column = 52;
	AngleSum(x, y, img.Width, img.Height, blockSize, orientationArrayLength, img, oriented, orient16);

	/*if (abs(angleSum - (float)PI) < 0.1f || abs(angleSum + (float)PI) < 0.1f || abs(angleSum - 2.0f * (float)PI) < 0.1f || abs(angleSum + 2.0f*(float)PI) < 0.1f)
	{
		dev_img.SetAt(row, column, 255);
	}
	else
	{
		dev_img.SetAt(row, column, 0);
	}*/ //temporary commented

}

void PoincareDetect(int* src, int width, int height, int blockSize, float* orientated, int* dist)
{
	cudaSetDevice(0);
	cudaError_t error = cudaGetLastError();

	CUDAArray<int> img = CUDAArray<int>(src, width, height);
//	CUDAArray<int> dev_img = CUDAArray<int>( width, height);
	//OrientationFieldInPixels(orientated, (float*)src, width, height);		//взяли поле ориентаций
	error = cudaGetLastError();

	int orientationArrayLength = 16;

	//dim3 gridSize = dim3(ceilMod(width, defaultThreadCount), ceilMod(height, defaultThreadCount));
	//dim3 blocksSize = dim3(defaultThreadCount, defaultThreadCount);

	CUDAArray<float> cudaOriented = CUDAArray<float>(orientated, width, height);
	//error = cudaGetLastError();
	float* orient16Float = new float[16];
	CUDAArray<float> orient16 = CUDAArray<float>(16, 1);
	float* tmpOrient = new float[364 * 256];
	cudaOriented.GetData(tmpOrient);
	Detect <<< 1, 1 >>> (52, 52, img,  blockSize, cudaOriented, orientationArrayLength, orient16);
	error = cudaGetLastError();
	orient16.GetData(orient16Float);
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 364; j++)
		{
			continue;
		}
		
	}

	for (int i = 0; i < 16; i++)
	{
		continue;
	}

	error = cudaGetLastError();
	orient16.Dispose();
	//dev_img.GetData(dist);
	error = cudaGetLastError();
	img.Dispose();
	//dev_img.Dispose();
	error = cudaGetLastError();
	cudaOriented.Dispose();
	cudaDeviceReset();
}

int main()
{
	int width=256, height=364;
	int* bmp = loadBmp("1.bmp", &width, &height);
	int* dist = new int[width* height];
	double* oriented = new  double[width*height];
	int k = 364 * 256;
	FILE* file = fopen("D:\\Temp.bin", "rb");
	int reallyRead = fread(oriented, sizeof(double),  k, file); 

	fclose(file);

	float* oriented2 = new  float[width, height];
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{			// уточнить координаты с учётом поворота пиксельвайза, вродь так
			oriented2[i, j] = (float)oriented[i*width+ j]; // потому то цпушный пикселвайз возвращает транспонированную матрицу оринетации
		}
	}
	

	

	PoincareDetect(bmp, width, height, 5, oriented2, dist);
	
	saveBmp("res.bmp", dist, width, height);

    return 0;
}
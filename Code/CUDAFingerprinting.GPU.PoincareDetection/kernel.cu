
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
#define FieldSize 5
#define OrientationArrayLength() (FieldSize*FieldSize - (FieldSize-2)*(FieldSize-2))
using namespace std;

extern "C"
{
	__declspec(dllexport)  void PoincareDetect(int* src, int width, int height, float* orientated, int* dist);
}


__device__  float AngleSum(int x, int y, int width, int height,  CUDAArray<float> oriented)
{
	float value;
	float OrrArr[OrientationArrayLength()];
	float angleSum;
	
	for (short i = 0; i < OrientationArrayLength() / 4; i++)
	{
		int left = x - FieldSize / 2;
		int right = y + i - FieldSize / 2;
		if (x - FieldSize / 2 >= 0 && y + i - FieldSize / 2 >= 0 && y + i - FieldSize / 2 < height && x - FieldSize / 2 < width)
		{
			value = oriented.At(left, right);
			OrrArr[i] = value;
		}
		else
		{
			OrrArr[i] = 0.0f;
		}
	}

	for (short i = OrientationArrayLength() / 4; i < OrientationArrayLength() / 2; i++)
	{
		int left = x - FieldSize / 2 + i - OrientationArrayLength() / 4;
		int right = y + FieldSize / 2;
		if (left >= 0 && right >= 0 && right < height && left < width)
		{
			value = oriented.At(left, right);
			OrrArr[i] = value;
		}
		else
		{
			OrrArr[i] = 0.0f;
		}
	}

	for (short i = OrientationArrayLength() / 2; i < OrientationArrayLength() / 2 + OrientationArrayLength() / 4; i++)
	{
		int left = x + FieldSize / 2;
		int right = y - FieldSize / 2 - (i - 3 * OrientationArrayLength() / 4);
		if (left >= 0 && right >= 0 && right < height && left < width)
		{
			value = oriented.At(left, right);
			OrrArr[i] = value;
		}
		else
		{
			OrrArr[i] = 0.0f;
		}
	}

	for (short i = OrientationArrayLength() / 4 + OrientationArrayLength() / 2; i < OrientationArrayLength(); i++)
	{
		int left = x - FieldSize / 2 - (i - OrientationArrayLength());
		int right = y - FieldSize / 2;
		if (left >= 0 && right >= 0 && right < height && left < width)
		{
			value = oriented.At(left, right);
			OrrArr[i] = value;
		}
		else
		{
			OrrArr[i] = 0.0f;
		}
	}
	angleSum = 0.0f;
	for (short i = 0; i < 16; i++)
	{

		if (abs(-OrrArr[i] + OrrArr[(i + 1) % OrientationArrayLength()]) <
			abs((float)PI + (-OrrArr[i] + OrrArr[(i + 1) % OrientationArrayLength()])))
		{
			angleSum += -OrrArr[i] + OrrArr[(i + 1) % OrientationArrayLength()];
		}
		else
		{
			angleSum += (float)PI + (-OrrArr[i] + OrrArr[(i + 1) % OrientationArrayLength()]);
		}
	}
	return angleSum;
}

__global__ void Detect(CUDAArray<int> dev_img,  CUDAArray<float> oriented)
{
	int column = defaultColumn();
	int row = defaultRow();
	float angleSum;

	if (row < dev_img.Height && column < dev_img.Width && row >= 0 && column >=0)
	{
		angleSum = AngleSum(row, column, dev_img.Width, dev_img.Height, oriented);
	}
	else
	{
		angleSum = 0.0f;
	}
	if (abs(angleSum - (float)PI) < 0.001f || abs(angleSum + (float)PI) < 0.001f || abs(angleSum - 2.0f * (float)PI) < 0.001f)
	{
		dev_img.SetAt(row, column, 128);
	}
}

void PoincareDetect(int* src, int width, int height, float* orientated, int* dist)
{
	cudaSetDevice(0);

	CUDAArray<int> dev_img = CUDAArray<int>(src, width, height);

	dim3 gridSize = dim3(ceilMod(width, defaultThreadCount), ceilMod(height, defaultThreadCount));
	dim3 blocksSize = dim3(defaultThreadCount, defaultThreadCount);

	CUDAArray<float> cudaOriented = CUDAArray<float>(orientated, width, height);

	Detect <<< gridSize, blocksSize >>> (dev_img, cudaOriented);

	dev_img.GetData(dist);
	dev_img.Dispose();
	cudaOriented.Dispose();
	cudaDeviceReset();
}

int main()
{
	/* // Если запускать ехе с этим кодом, получится перевёрнутая карта сингулярностей
		// В шарповом тесте картинка получается верной
	int width=256, height=364;
	int* bmp = loadBmp("1.bmp", &width, &height);
	int* dist = new int[width* height];
	double* oriented = new  double[width*height];
	int k = 364 * 256;
	FILE* file = fopen("D:\\Temp.bin", "rb");
	int reallyRead = fread(oriented, sizeof(double),  k, file); 

	fclose(file);

	float* oriented2 = new  float[width* height];
	for (int i = 0; i < width*height; i++)
	{
		oriented2[i] = (float)oriented[i]; 
	}

	PoincareDetect(bmp, width, height, oriented2, dist);
	
	saveBmp("res.bmp", dist, width, height);*/

    return 0;
}
#define _USE_MATH_DEFINES 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDAArray.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "Filter.cuh"
#include "ImageLoading.cuh"
#include "OrientationField.cuh"
#include "math_constants.h"
extern "C"
{
	__declspec(dllexport) void Enhance(float* source, int imgWidth, int imgHeight, float* res, float* orientationMatrix,
		float frequency, int filterSize, int angleNum);
}
__global__ void EnhancePixel(CUDAArray<float> img, CUDAArray<float> result, CUDAArray<float> orientMatrix, float frequency,
	CUDAArray<float> filters, int angleNum, float* angles)
{
	int row = defaultRow();
	int column = defaultColumn();

	int filterSize  = filters.Height;
	int center      = filterSize / 2;
	int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;

	if (row < img.Height && column < img.Width) {
		float diff = FLT_MAX;
		int angle = 0;
		for (int angInd = 0; angInd < angleNum; angInd++)
		{
			if (abs(angles[angInd] - orientMatrix.At(row, column)) < diff)
			{
				angle = angInd;
				diff = abs(angles[angInd] - orientMatrix.At(row, column));
			}
		}

		float sum = 0;
		for (int drow = -upperCenter; drow <= center; drow++)
		{
			for (int dcolumn = -upperCenter; dcolumn <= center; dcolumn++)
			{
				float filterValue = filters.At(center - drow, filterSize* angle + (center - dcolumn));
				
				int indexRow = row + drow;
				int indexColumn = column + dcolumn;

				if (indexRow < 0)    indexRow    = 0;
				if (indexColumn < 0) indexColumn = 0;
				if (indexRow >= img.Height)   indexRow    = img.Height - 1;
				if (indexColumn >= img.Width) indexColumn = img.Width - 1;

				sum += filterValue * img.At(indexRow, indexColumn);
			}
		}
		//sum = (((int)sum) % 256 + (sum - ((int)sum)));//I would've written 'sum %= 256' if 'sum' was integer.
		if (sum < 0) sum = 0;
		result.SetAt(row, column, sum);
	}
}

void Enhance(float* source, int imgWidth, int imgHeight, float* res, float* orientationMatrix, 
	float frequency, int filterSize, int angleNum)
{
	CUDAArray<float> result       = CUDAArray<float>(imgWidth, imgHeight);
	CUDAArray<float> img          = CUDAArray<float>(source, imgWidth, imgHeight);
	CUDAArray<float> orientMatrix = CUDAArray<float>(orientationMatrix, imgWidth, imgHeight);

	float* angles = (float*)malloc(angleNum * sizeof(float));//passing small array is better than creating it multiple times, I think.
	const float constAngle = CUDART_PI_F / angleNum;
	for (int i = 0; i < angleNum; i++)
		angles[i] = constAngle * i - CUDART_PI_F / 2;
	float* dev_angles;
	cudaMalloc((void**)&dev_angles, angleNum * sizeof(float));
	cudaMemcpy(dev_angles, angles, angleNum * sizeof(float), cudaMemcpyHostToDevice);
	
	CUDAArray<float> filters = MakeGabor32Filters(angleNum, frequency);

	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize  = dim3(ceilMod(imgWidth, defaultThreadCount), ceilMod(imgHeight, defaultThreadCount));
	EnhancePixel << <gridSize, blockSize >> >(img, result, orientMatrix, frequency, filters, angleNum, dev_angles);
	result.GetData(res);
}

void main()
{
	int width;
	int height;
	char* filename = "..\\4_8.bmp";  //Write your way to bmp file
	int* img = loadBmp(filename, &width, &height);
	float* source = (float*)malloc(height*width*sizeof(float));
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			source[i * width + j] = (float)img[i * width + j];
		}
	float* b = (float*)malloc(height * width * sizeof(float));
	float* orMatr = OrientationFieldInPixels(source, width, height);
	Enhance(source, width, height, b, orMatr, (float)1 / 9, 32, 8);
	saveBmp("..\\res.bmp", b, width, height);

	free(source);
	free(img);
	free(b);
}

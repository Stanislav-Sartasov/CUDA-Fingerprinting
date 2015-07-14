#define _USE_MATH_DEFINES 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDAArray.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "Filter.cuh"
__global__ void EnhancePixel(CUDAArray<float> img, CUDAArray<float> result, CUDAArray<float> orientMatrix, float frequency,
	int filterSize, CUDAArray<float> filters, int angleNum, float* angles)
{
	int row = defaultRow();
	int column = defaultColumn();

	int center = filterSize / 2; //filter is always a square.
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

		int tX = threadIdx.x;
		int tY = threadIdx.y;
		__shared__ float filterCache[32 * 32];

		if (tX < filterSize && tY < filterSize) //what this condition is about?
		{
			int indexLocal = tX + tY * filterSize;
			filterCache[indexLocal] = filters.At(tY * angle, tX);
		}
		__syncthreads();

		float sum = 0;
		for (int drow = -center; drow <= center; drow++)
		{
			for (int dcolumn = -center; dcolumn <= center; dcolumn++)
			{
				float filterValue = filterCache[filterSize*(drow + center) + dcolumn + center];

				int indexRow = row + drow;
				int indexColumn = column + dcolumn;

				if (indexRow < 0 || indexRow >= img.Height || indexColumn < 0 || indexColumn >= img.Width)
					continue;

				float value = img.At(indexRow, indexColumn);
				sum += filterValue * value;
			}
		}
		//if (sum > 255) //is there a way to take a module of a floating-point number?
		//{
		//	sum = 255;
		//}
		sum = (((int)sum) % 256 + (sum - ((int)sum)));//this IS the elegant way, isn't it?
		result.SetAt(row, column, sum);
	}
}

void Enhance(CUDAArray<float> img, float* res, CUDAArray<float> orientMatrix, float frequency, int filterSize,
	int angleNum)
{
	CUDAArray<float> result = CUDAArray<float>(img.Height, img.Width);

	float* angles = (float*) malloc(angleNum * sizeof(float));
	const float constAngle = M_PI / angleNum;
	for (int i = 0; i < angleNum; i++)
		angles[i] = constAngle * i - M_PI / 2;
	float* dev_angles;
	cudaMalloc((void**)&angles, angleNum * sizeof(float));
	cudaMemcpy(dev_angles, angles, angleNum * sizeof(float), cudaMemcpyHostToDevice);

	float* filter = (float*)malloc(filterSize * (filterSize * angleNum) * sizeof(float));
	MakeGabor16Filters(filter, angleNum, frequency);//replace with proper filter-getting function
	CUDAArray<float> filters = CUDAArray<float>(filter, filterSize, filterSize * angleNum);

	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(img.Width, defaultThreadCount), ceilMod(img.Height, defaultThreadCount));
	EnhancePixel<<<gridSize, blockSize>>>(img, result, orientMatrix, frequency, filterSize, filters, angleNum, angles);
	result.GetData(res);
}
//__global__ void Enhance(CUDAArray<float> img, float* res, CUDAArray<float> orientMatrix, float frequency, int filterSize,
//	int angleNum)
//{
//	int imgHeight = img.Height;
//	int imgWidth = img.Width;
//	CUDAArray<float> result = CUDAArray<float>(imgHeight, imgWidth);
//	float* angles = (float*)malloc(angleNum * sizeof(float));
//	const float constAngle = M_PI / angleNum;
//	for (int i = 0; i < angleNum; i++)
//		angles[i] = constAngle * i - M_PI / 2;
//
//	var gabor = new GaborFilter(angleNum, filterSize, frequency);
//	int center = filterSize / 2; //filter is always a square.
//	int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;
//
//	for (int i = 0; i < imgHeight; i++)
//	{
//		for (int j = 0; j < imgWidth; j++)
//		{
//			float diff = FLT_MAX;
//			int angle = 0;
//			for (int angInd = 0; angInd < angleNum; angInd++)
//			if (abs(angles[angInd] - orientMatrix.At(i, j)) < diff)
//			{-
//				angle = angInd;
//				diff = abs(angles[angInd] - orientMatrix.At(i, j));
//			}
//			for (int u = -upperCenter; u <= center; u++)
//			{
//				for (int v = -upperCenter; v <= center; v++)
//				{
//					int indexX = i + u;
//					int indexY = j + v;
//					if (indexX < 0) indexX = 0;
//					if (indexX >= imgHeight) indexX = imgHeight - 1;
//					if (indexY < 0) indexY = 0;
//					if (indexY >= imgWidth) indexY = imgWidth - 1;
//					result.SetAt(i, j, result.At(i, j) + gabor.Filters[angle].Matrix[center - u, center - v] * img.At(indexX, indexY));
//				}
//			}
//			if (result.At(i, j) > 255)
//			{
//				result.SetAt(i, j, 255);
//			}
//		}
//	}
//	res = result.GetData();
//}
//void main()
//{
//
//}
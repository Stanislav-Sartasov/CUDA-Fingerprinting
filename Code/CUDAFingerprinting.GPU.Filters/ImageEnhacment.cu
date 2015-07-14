#define _USE_MATH_DEFINES 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDAArray.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
__global__ void EnhancePixel(CUDAArray<float> img, float* res, CUDAArray<float> orientMatrix, float frequency, int filterSize,
	int angleNum)
{
	int imgHeight = img.Height;
	int imgWidth = img.Width;
	int row = defaultRow();
	int column = defaultColumn();
	CUDAArray<float> result = CUDAArray<float>(imgHeight, imgWidth);
	float* angles = (float*)malloc(angleNum * sizeof(float));
	const float constAngle = M_PI / angleNum;
	for (int i = 0; i < angleNum; i++)
		angles[i] = constAngle * i - M_PI / 2;

	var gabor = new GaborFilter(angleNum, filterSize, frequency);
	int center = filterSize / 2; //filter is always a square.
	int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;

	for (int i = 0; i < imgHeight; i++)
	{
		for (int j = 0; j < imgWidth; j++)
		{
			float diff = FLT_MAX;
			int angle = 0;
			for (int angInd = 0; angInd < angleNum; angInd++)
			if (abs(angles[angInd] - orientMatrix.At(i, j)) < diff)
			{
				angle = angInd;
				diff = abs(angles[angInd] - orientMatrix.At(i, j));
			}
			for (int u = -upperCenter; u <= center; u++)
			{
				for (int v = -upperCenter; v <= center; v++)
				{
					int indexX = i + u;
					int indexY = j + v;
					if (indexX < 0) indexX = 0;
					if (indexX >= imgHeight) indexX = imgHeight - 1;
					if (indexY < 0) indexY = 0;
					if (indexY >= imgWidth) indexY = imgWidth - 1;
					result.SetAt(i, j, result.At(i, j) + gabor.Filters[angle].Matrix[center - u, center - v] * img.At(indexX, indexY));
				}
			}
			if (result.At(i, j) > 255)
			{
				result.SetAt(i, j, 255);
			}
		}
	}
	res = result.GetData();
}
__global__ void Enhance(CUDAArray<float> img, float* res, CUDAArray<float> orientMatrix, float frequency, int filterSize,
	int angleNum)
{
	int imgHeight = img.Height;
	int imgWidth = img.Width;
	CUDAArray<float> result = CUDAArray<float>(imgHeight, imgWidth);
	float* angles = (float*)malloc(angleNum * sizeof(float));
	const float constAngle = M_PI / angleNum;
	for (int i = 0; i < angleNum; i++)
		angles[i] = constAngle * i - M_PI / 2;

	var gabor = new GaborFilter(angleNum, filterSize, frequency);
	int center = filterSize / 2; //filter is always a square.
	int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;

	for (int i = 0; i < imgHeight; i++)
	{
		for (int j = 0; j < imgWidth; j++)
		{
			float diff = FLT_MAX;
			int angle = 0;
			for (int angInd = 0; angInd < angleNum; angInd++)
			if (abs(angles[angInd] - orientMatrix.At(i, j)) < diff)
			{
				angle = angInd;
				diff = abs(angles[angInd] - orientMatrix.At(i, j));
			}
			for (int u = -upperCenter; u <= center; u++)
			{
				for (int v = -upperCenter; v <= center; v++)
				{
					int indexX = i + u;
					int indexY = j + v;
					if (indexX < 0) indexX = 0;
					if (indexX >= imgHeight) indexX = imgHeight - 1;
					if (indexY < 0) indexY = 0;
					if (indexY >= imgWidth) indexY = imgWidth - 1;
					result.SetAt(i, j, result.At(i, j) + gabor.Filters[angle].Matrix[center - u, center - v] * img.At(indexX, indexY));
				}
			}
			if (result.At(i, j) > 255)
			{
				result.SetAt(i, j, 255);
			}
		}
	}
	res = result.GetData();
}
void main()
{

}
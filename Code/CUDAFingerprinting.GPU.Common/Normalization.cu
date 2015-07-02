#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>
#include "CUDAArray.cuh"
float CalculateMean(CUDAArray<float> image)
{
	int height = image.Height;
	int width = image.Width;
	float mean = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			mean += image.At(i, j) / (height * width);
		}
	}
	return mean;
}

float CalculateVariation(CUDAArray<float> image, float mean)
{
	int height = image.Height;
	int width = image.Width;
	float variation = 0;
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			variation += pow((image.At(i, j) - mean), 2) / (height * width);
		}
	}
	return variation;
}
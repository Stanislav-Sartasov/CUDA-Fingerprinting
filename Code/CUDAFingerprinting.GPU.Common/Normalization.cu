#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include <math.h>
#include "CUDAArray.cuh"

double CalculateMean(CUDAArray<float> image)
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

CUDAArray<float> DoNormalization(CUDAArray<float> image, int bordMean, int bordVar)
{
	float mean = CalculateMean(image);
	float variation = CalculateVariation(image, mean);

	for (int i = 0; i < image.Width; i++)
	{
		for (int j = 0; j < image.Height; j++)
		{
			if (image.At(i, j) > mean)
			{
				image.SetAt(i, j, bordMean + sqrt((bordVar * pow(image.At(i, j) - mean, 2)) / variation));
			}
			else
			{
				image.SetAt(i, j), bordMean - sqrt((bordVar * pow(image.At(i, j) - mean, 2)) / variation));
			}
		}
	}

	return image;
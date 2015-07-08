#ifndef CUDAFINGEROPRINTING_THINNERUTILS
#define CUDAFINGEROPRINTING_THINNERUTILS

#include <stdlib.h>

const double BLACK = 0.0;
const double WHITE = 255.0;

enum PixelType
{
	FILLED,
	EMPTY,
	ANY,
	CENTER,
	AT_LEAST_ONE_EMPTY
};

double** copy2DArray(double** source, int width, int height)
{
	double** nA = (double**)malloc(sizeof(double*) * height);
	for (int i = 0; i < height; i++)
	{
		nA[i] = (double*)malloc(sizeof(double) * width);
		for (int j = 0; j < width; j++)
		{
			nA[i][j] = source != NULL ? source[i][j] : 0.0;
		}
	}
	return nA;
}

void free2DArray(double** arr, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		free(arr[i]);
	}
	free(arr);
}

void free2DArray(PixelType** arr, int width, int height)
{
	for (int i = 0; i < height; i++)
	{
		free(arr[i]);
	}
	free(arr);
}

#endif
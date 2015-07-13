#ifndef CUDAFINGEROPRINTING_THINNERUTILS
#define CUDAFINGEROPRINTING_THINNERUTILS

#include <stdlib.h>

#ifndef BLACK
#define BLACK 0.0
#endif

#ifndef WHITE
#define WHITE 255.0
#endif

enum PixelType
{
	FILLED,
	EMPTY,
	ANY,
	CENTER,
	AT_LEAST_ONE_EMPTY
};

double* copy1DArray(double* source, int size)
{
	double* nA = (double*)malloc(sizeof(double) * size);
	for (int i = 0; i < size; i++)
	{
		nA[i] = source != NULL ? source[i] : 0.0;
	}
	return nA;
}

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

double* copy2DArrayTo1D(double** source, int width, int height)
{
	double* nA = (double*)malloc(sizeof(double) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			nA[i * width + j] = source != NULL ? source[i][j] : 0.0;
		}
	}
	return nA;
}

double** copy1DArrayTo2D(double* source, int width, int height)
{
	double** nA = (double**)malloc(sizeof(double*) * height);
	for (int i = 0; i < height; i++)
	{
		nA[i] = (double*)malloc(sizeof(double) * width);
		for (int j = 0; j < width; j++)
		{
			nA[i][j] = source != NULL ? source[i * width + j] : 0.0;
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
#ifndef CUDAFINGEROPRINTING_UTILS
#define CUDAFINGEROPRINTING_UTILS

/*Test helpers*/

#include <stdio.h>
#include <stdlib.h>

double** intToDoubleArray(int* input, int width, int height)
{
	double** output = (double**)malloc(sizeof(double*) * height);
	for (int i = 0; i < height; i++)
	{
		output[i] = (double*)malloc(sizeof(double) * width);
		for (int j = 0; j < width; j++)
		{
			output[i][j] = input[i * width + j] > 129 ?
				255.0 :
				input[i * width + j] < 127 ?
				0.0 :
				128.0;
		}
	}
	return output;
}

int* doubleToIntArray(double** input, int width, int height)
{
	int* output = (int*)malloc(sizeof(int) * height * width);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			output[i * width + j] = input[i][j] > 129.0 ?
				255 :
				input[i][j] < 127.0 ?
				0 :
				128;
		}
	}
	return output;
}

void WriteArrayPic(double** bytes, int h, int w)
{
	for (int x = 0; x < h; x++)
	{
		for (int y = 0; y < w; y++)
		{
			printf("%c", bytes[h - 1 - x][y] != 0 ? '=' : '+');
		}
		printf("\n");
	}
}

//overlaps skeleton above background
double** OverlapArrays(double** skeleton, double** background, int width, int height)
{
	double** output = (double**)malloc(sizeof(double*) * height);
	for (int i = 0; i < height; i++)
	{
		output[i] = (double*)malloc(sizeof(double) * width);
		for (int j = 0; j < width; j++)
		{
			output[i][j] = skeleton[i][j] < 250.0 ? 128.0 : background[i][j];
		}
	}
	return output;
}

#endif
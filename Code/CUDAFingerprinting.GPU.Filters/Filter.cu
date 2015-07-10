#include "CUDAArray.cuh"
#include <stdlib.h>

#define _USE_MATH_DEFINES
#include <math.h>

CUDAArray<float> Filter(int size, float angle, float frequency)
{
	float* matrix = (float*)malloc(size * size * sizeof(float));

	float aCos = cos(M_PI / 2 + angle);
	float aSin = sin(M_PI / 2 + angle);

	int center = size / 2;
	int upperCenter = (size & 1) == 1 ? center - 1 : center;

	for (int i = -upperCenter; i < center; i++)
	{
		for (int j = -upperCenter; j < center; j++)
		{
			matrix[i * size + j] = exp(-0.5 * ((i * aSin + j * aCos) * (i * aSin + j * aCos) / 16 + (-i *aCos + j * aSin) * (-i *aCos + j * aSin) / 16)) * cos(2 * M_PI * (i * aSin + j * aCos) * frequency);
		}
	}

	return CUDAArray<float>(matrix, size, size);
}
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "Convolution.cuh"
#include "constsmacros.h"
#include <float.h>
#include "CUDAArray.cuh"

void OrientationField(CUDAArray<float> source, int sizeX, int sizeY){
	const int DefaultBlockSize = 16;
	CUDAArray<float> Orientation(sizeY, sizeX);

	// фильтры Собеля
	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	// фильтры для девайса
	CUDAArray<float> filterX(filterXLinear, 3, 3);
	CUDAArray<float> filterY(filterYLinear, 3, 3);

	// Градиенты
	CUDAArray<float> Gx;
	CUDAArray<float> Gy;
	Convolve(Gx, source, filterX);
	Convolve(Gy, source, filterX);

	// проход по всем блокам
	/*for (int i = 0; i < sizeY / DefaultBlockSize; i++){
	for (int j = 0; j < sizeX / DefaultBlockSize; j++){
	double numerator = 0;
	double denominator = 0;
	for (int x = 0; x < DefaultBlockSize; x++)
	{
	for (int y = 0; y < DefaultBlockSize; y++)
	{
	numerator += Gx[x * DefaultBlockSize + y] * Gy[x, y];
	denominator += Gx[i, j] * Gx[i, j] - Gy[i, j] * Gy[i, j];
	}
	}
	if (denominator == 0)
	{
	_orientation = Math.PI / 2;
	}
	else
	{
	_orientation = Math.PI / 2 + Math.Atan2(2 * numerator, denominator) / 2;
	if (_orientation > Math.PI / 2) _orientation -= Math.PI;
	}
	}
	}*/


}


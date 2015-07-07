#define _USE_MATH_DEFINES 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "Convolution.cuh"
#include "constsmacros.h"
#include "CUDAArray.cuh"
/////////////////////////////////////////// НОВЫЙ ///////////////////////////////////////////
__global__ void cudaSetOrientation(CUDAArray<float> orientation, CUDAArray<float> gradientX, CUDAArray<float> gradientY){
	float numerator;
	float denominator;

	int column = defaultColumn();
	int row = defaultRow();
	int threadColumn = threadIdx.x;
	int threadRow = threadIdx.y;
	float GyValue = gradientY.At(row, column);
	float GxValue = gradientX.At(row, column);

	// вычисление числителя и знаменателя
	// сначала перемножаем соответствующие элементы матрицы, результат помещаем в shared память
	__shared__ float product[16][16];
	__shared__ float sqrdiff[16][16];

	product[threadRow][threadColumn] = GxValue * GyValue; // копируем в общую память произведения соответствующих элементов 
	sqrdiff[threadRow][threadColumn] = GxValue * GxValue - GyValue * GyValue; // разность квадратов
	__syncthreads();  // ждем пока все нити сделают вычисления

	// теперь нужно просуммировать элементы матриц
	// суммируем элементы строк, результаты суммы будут в первой колонке 
	for (int s = blockDim.x / 2; s > 0; s = s / 2) {		// суммируем так, чтобы нити не работали с одной и той же памятью
		if (threadColumn < s) {
			product[threadRow][threadColumn] += product[threadRow][threadColumn + s];
			sqrdiff[threadRow][threadColumn] += sqrdiff[threadRow][threadColumn + s];
		}
		__syncthreads();
	}
	// суммируем элементы первого столбца, получаем сумму сумм
	if (threadColumn == 0){
		for (int s = blockDim.y / 2; s > 0; s = s / 2) {		// суммируем так, чтобы нити не работали с одной и той же памятью
			if (threadRow < s) {
				product[threadRow][threadColumn] += product[threadRow + s][threadColumn];
				sqrdiff[threadRow][threadColumn] += sqrdiff[threadRow + s][threadColumn];
			}
			__syncthreads();
		}
	}

	// после прохода циклов результаты будут находится в находиться в product[0][0] и sqrdiff[0][0]
	numerator = 2 * product[0][0];
	denominator = sqrdiff[0][0];

	// определяем значение угла ориентации
	if (denominator == 0){
		orientation.SetAt(row, column, M_PI_2);
	}
	else{
		orientation.SetAt(row, column, M_PI_2 + atan2(2 * numerator, denominator) / 2.0);
		if (orientation.At(row, column) > M_PI_2){
			orientation.SetAt(row, column, orientation.At(row, column) - M_PI);
		}
	}
}

void SetOrientation(CUDAArray<float> orientation, CUDAArray<float> source, int defaultBlockSize, CUDAArray<float> gradientX, CUDAArray<float> gradientY){
	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize =
		dim3(ceilMod(source.Width, defaultBlockSize),
		ceilMod(source.Height, defaultBlockSize));
	cudaSetOrientation << <gridSize, blockSize >> >(orientation, gradientX, gradientY);
}

void OrientationField(float* floatArray, int width, int height){
	CUDAArray<float> source(floatArray, width, height);
	const int defaultBlockSize = 16;
	CUDAArray<float> Orientation(source.Width, source.Height);

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

	// вычисляем направления
	SetOrientation(Orientation, source, defaultBlockSize, Gx, Gy);
}


#define _USE_MATH_DEFINES 
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cuda_runtime.h"
#include <device_functions.h>
#include "device_launch_parameters.h"
#include "Convolution.cuh"
#include "constsmacros.h"
#include "imageLoading.cuh"
#include "CUDAArray.cuh"
#include "OrientationField.cuh"

// ----------- GPU ----------- //

__global__ void cudaSetOrientationInPixels(CUDAArray<float> orientation, CUDAArray<float> gradientX, CUDAArray<float> gradientY){
	int centerRow = defaultRow();
	int centerColumn = defaultColumn();

	const int size = 16;
	const int center = size / 2;
	const int upperLimit = center - 1;
	
	float product[size][size];
	float sqrdiff[size][size];

	for (int i = -center; i <= upperLimit; i++){
		for (int j = -center; j <= upperLimit; j++){
			if (i + centerRow < 0 || i + centerRow > gradientX.Height || j + centerColumn < 0 || j + centerColumn > gradientX.Width){		// выход за пределы картинки
				product[i + center][j + center] = 0;
				sqrdiff[i + center][j + center] = 0;
			}
			else{
				float GxValue = gradientX.At(i + centerRow, j + centerColumn);
				float GyValue = gradientY.At(i + centerRow, j + centerColumn);
				product[i + center][j + center] = GxValue * GyValue;						// поэлементное произведение
				sqrdiff[i + center][j + center] = GxValue * GxValue - GyValue * GyValue;	// разность квадратов
			}
		}
	}
	__syncthreads();  // ждем пока все нити сделают вычисления

	float numerator = 0;
	float denominator = 0;
	// вычисление сумм
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++){
			numerator += product[i][j];
			denominator += sqrdiff[i][j];
		}
	}
	__syncthreads();

	// определяем значение угла ориентации
	if (denominator == 0){
		orientation.SetAt(centerRow, centerColumn, M_PI_2);
	}
	else{
		orientation.SetAt(centerRow, centerColumn, M_PI_2 + atan2(2 * numerator, denominator) / 2.0f);
		if (orientation.At(centerRow, centerColumn) > M_PI_2)
		{
			float index = orientation.At(centerRow, centerColumn) - M_PI;
			orientation.SetAt(centerRow, centerColumn, index);
		}
	}
}

__global__ void cudaSetOrientationInBlocks(CUDAArray<float> orientation, CUDAArray<float> gradientX, CUDAArray<float> gradientY){
	float numerator;
	float denominator;

	int column = defaultColumn();			// глобальные индексы -- индексы пикселя на целой картинке
	int row = defaultRow();
	int threadColumn = threadIdx.x;			// локальные индексы -- индексы пикселя в блоке 
	int threadRow = threadIdx.y;
	float GyValue = gradientY.At(row, column);
	float GxValue = gradientX.At(row, column);

	const int defaultBlockSize = 16;		// размер блока, по которому считается направление

	// вычисление числителя и знаменателя
	// сначала перемножаем соответствующие элементы матрицы, результат помещаем в shared память
	__shared__ float product[defaultBlockSize][defaultBlockSize];
	__shared__ float sqrdiff[defaultBlockSize][defaultBlockSize];

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
	numerator = product[0][0];
	denominator = sqrdiff[0][0];

	// определяем значение угла ориентации
	if (denominator == 0){
		orientation.SetAt(row, column, M_PI_2);
	}
	else{
		orientation.SetAt(row, column, M_PI_2 + atan2(2 * numerator, denominator) / 2.0f);
		if (orientation.At(row, column) > M_PI_2){
			orientation.SetAt(row, column, orientation.At(row, column) - M_PI);
		}
	}
}

// ----------- CPU ----------- //

void SetOrientationInBlocks(CUDAArray<float> orientation, CUDAArray<float> source, const int defaultBlockSize, CUDAArray<float> gradientX, CUDAArray<float> gradientY){
	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize =
		dim3(ceilMod(source.Width, defaultBlockSize),
		ceilMod(source.Height, defaultBlockSize));
	cudaSetOrientationInBlocks << <gridSize, blockSize >> >(orientation, gradientX, gradientY);
	cudaError_t error = cudaDeviceSynchronize();
}

void SetOrientationInPixels(CUDAArray<float> orientation, CUDAArray<float> source, CUDAArray<float> gradientX, CUDAArray<float> gradientY){
	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize =
		dim3(ceilMod(source.Width, defaultThreadCount),
		ceilMod(source.Height, defaultThreadCount));
	cudaSetOrientationInPixels << <gridSize, blockSize >> >(orientation, gradientX, gradientY);
	cudaError_t error = cudaDeviceSynchronize();
	float* o = orientation.GetData();
}


float* OrientationFieldInBlocks(float* floatArray, int width, int height){
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
	CUDAArray<float> Gx(width, height);
	CUDAArray<float> Gy(width, height);
	Convolve(Gx, source, filterX);
	Convolve(Gy, source, filterY);

	// вычисляем направления
	SetOrientationInBlocks(Orientation, source, defaultBlockSize, Gx, Gy);
	
	return Orientation.GetData();
}

float* OrientationFieldInPixels(float* floatArray, int width, int height){

	CUDAArray<float> source(floatArray, width, height);
	CUDAArray<float> Orientation(source.Width, source.Height);

	// фильтры Собеля
	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	// фильтры для девайса
	CUDAArray<float> filterX(filterXLinear, 3, 3);
	CUDAArray<float> filterY(filterYLinear, 3, 3);

	// Градиенты
	CUDAArray<float> Gx(width, height);
	CUDAArray<float> Gy(width, height);
	Convolve(Gx, source, filterX);
	Convolve(Gy, source, filterY);

	SetOrientationInPixels(Orientation, source, Gx, Gy);
	
	return Orientation.GetData();
}

void OrientationFieldInPixels(float* res, float* floatArray, int width, int height){

	CUDAArray<float> source(floatArray, width, height);
	CUDAArray<float> Orientation(source.Width, source.Height);

	// фильтры Собеля
	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	// фильтры для девайса
	CUDAArray<float> filterX(filterXLinear, 3, 3);
	CUDAArray<float> filterY(filterYLinear, 3, 3);

	// Градиенты
	CUDAArray<float> Gx(width, height);
	CUDAArray<float> Gy(width, height);
	Convolve(Gx, source, filterX);
	Convolve(Gy, source, filterY);

	SetOrientationInPixels(Orientation, source, Gx, Gy);

	Orientation.GetData(res);
}



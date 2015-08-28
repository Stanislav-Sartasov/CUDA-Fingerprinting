#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <malloc.h>
#include <stdlib.h>
#include <time.h>

#ifndef PI
#define PI 3.1415926535897932384626433832795f
#endif

#define defaultRow() blockIdx.y*blockDim.y + threadIdx.y
#define defaultColumn() blockIdx.x*blockDim.x + threadIdx.x
#define ceilMod(x ,y) (x + y - 1)/(y)

#define checkStatus() if (cudaStatus != cudaSuccess) goto Error

const int defaultThreadsCount = 16;

__device__ void countOptimumlPhiDEVICE(float* phi0, float sProd, float vProd, int maxIterations, float e)
{
	float resultPhi = *phi0;

	float cos_phi;
	float sin_phi;
	float newPhi;
	for (int i = 0; i < maxIterations; i++)
	{
		cos_phi = cosf(resultPhi);
		sin_phi = sinf(resultPhi);

		newPhi = resultPhi - (sin_phi * sProd + cos_phi * vProd) / (cos_phi * sProd - sin_phi * vProd);

		if (fabsf(newPhi - resultPhi) < e)
			break;
		else
			resultPhi = newPhi;
	}

	*phi0 = resultPhi;
}

__device__ void moveTriangleToStartDEVICE(float* ABC_A_x, float* ABC_A_y, float* ABC_B_x, float* ABC_B_y, float* ABC_C_x, float* ABC_C_y, float* dx, float* dy)
{
	*dx = ((*ABC_A_x) + (*ABC_B_x) + (*ABC_C_x)) / 3;
	*dy = ((*ABC_A_y) + (*ABC_B_y) + (*ABC_C_y)) / 3;

	*ABC_A_x -= (*dx); *ABC_B_x -= (*dx); *ABC_C_x -= (*dx);
	*ABC_A_y -= (*dy); *ABC_B_y -= (*dy); *ABC_C_y -= (*dy);
}

__device__ void countOptimumTransformationOfMovedTriangleABCtoABCDEVICE(
	float _ABC_A_x, float _ABC_A_y, float _ABC_B_x, float _ABC_B_y, float _ABC_C_x, float _ABC_C_y,
	float ABC_A_x, float ABC_A_y, float ABC_B_x, float ABC_B_y, float ABC_C_x, float ABC_C_y,
	float maxIterations, float e, int parts, float step,
	float* cos_phi, float* sin_phi, float* distance)
{
	//На входе имеем лишь сдвинутые треугольники

	float sProd = ABC_A_x * _ABC_A_x + ABC_A_y * _ABC_A_y + ABC_B_x * _ABC_B_x + ABC_B_y * _ABC_B_y + ABC_C_x * _ABC_C_x + ABC_C_y * _ABC_C_y;
	float vProd = ABC_A_x * _ABC_A_y - ABC_A_y * _ABC_A_x + ABC_B_x * _ABC_B_y - ABC_B_y * _ABC_B_x + ABC_C_x * _ABC_C_y - ABC_C_y * _ABC_C_x;

	*cos_phi = 1; *sin_phi = 0;
	*distance =
		(ABC_A_x - _ABC_A_x)*(ABC_A_x - _ABC_A_x) + (ABC_A_y - _ABC_A_y)*(ABC_A_y - _ABC_A_y) +
		(ABC_B_x - _ABC_B_x)*(ABC_B_x - _ABC_B_x) + (ABC_B_y - _ABC_B_y)*(ABC_B_y - _ABC_B_y) +
		(ABC_C_x - _ABC_C_x)*(ABC_C_x - _ABC_C_x) + (ABC_C_y - _ABC_C_y)*(ABC_C_y - _ABC_C_y);


	float optimumPhi;
	float optimum_cos;
	float optimum_sin;
	float tmp_distance;

	float new_ABC_A_x; float new_ABC_B_x; float new_ABC_C_x;
	float new_ABC_A_y; float new_ABC_B_y; float new_ABC_C_y;

	for (int i = 0; i < parts; i++)
	{
		optimumPhi = step * i - PI;

		countOptimumlPhiDEVICE(&optimumPhi, sProd, vProd, maxIterations, e);
		optimum_cos = cosf(optimumPhi);
		optimum_sin = sinf(optimumPhi);

		new_ABC_A_x = _ABC_A_x * optimum_cos - _ABC_A_y * optimum_sin;
		new_ABC_A_y = _ABC_A_x * optimum_sin + _ABC_A_y * optimum_cos;
		new_ABC_B_x = _ABC_B_x * optimum_cos - _ABC_B_y * optimum_sin;
		new_ABC_B_y = _ABC_B_x * optimum_sin + _ABC_B_y * optimum_cos;
		new_ABC_C_x = _ABC_C_x * optimum_cos - _ABC_C_y * optimum_sin;
		new_ABC_C_y = _ABC_C_x * optimum_sin + _ABC_C_y * optimum_cos;

		tmp_distance =
			(ABC_A_x - new_ABC_A_x)*(ABC_A_x - new_ABC_A_x) + (ABC_A_y - new_ABC_A_y)*(ABC_A_y - new_ABC_A_y) +
			(ABC_B_x - new_ABC_B_x)*(ABC_B_x - new_ABC_B_x) + (ABC_B_y - new_ABC_B_y)*(ABC_B_y - new_ABC_B_y) +
			(ABC_C_x - new_ABC_C_x)*(ABC_C_x - new_ABC_C_x) + (ABC_C_y - new_ABC_C_y)*(ABC_C_y - new_ABC_C_y);

		if (tmp_distance < *distance)
		{
			*cos_phi = optimum_cos;
			*sin_phi = optimum_sin;
			*distance = tmp_distance;
		}
	}
}

__device__ void countOptimumTransformationOfMovedTriangleDEVICE(
	float _ABC_A_x, float _ABC_A_y, float _ABC_B_x, float _ABC_B_y, float _ABC_C_x, float _ABC_C_y,
	float ABC_A_x, float ABC_A_y, float ABC_B_x, float ABC_B_y, float ABC_C_x, float ABC_C_y,
	float maxIterations, float e, int parts, float step,
	float* cos_phi, float* sin_phi, float* distance)
{
	float optimum_cos;
	float optimum_sin;
	float optimum_distance;

	countOptimumTransformationOfMovedTriangleABCtoABCDEVICE(
		_ABC_A_x, _ABC_A_y, _ABC_B_x, _ABC_B_y, _ABC_C_x, _ABC_C_y,
		ABC_A_x, ABC_A_y, ABC_B_x, ABC_B_y, ABC_C_x, ABC_C_y,
		maxIterations, e, parts, step,
		cos_phi, sin_phi, distance);

	countOptimumTransformationOfMovedTriangleABCtoABCDEVICE(
		_ABC_A_x, _ABC_A_y, _ABC_B_x, _ABC_B_y, _ABC_C_x, _ABC_C_y,
		ABC_B_x, ABC_B_y, ABC_C_x, ABC_C_y, ABC_A_x, ABC_A_y,
		maxIterations, e, parts, step,
		&optimum_cos, &optimum_sin, &optimum_distance);

	if (optimum_distance < *distance)
	{
		*cos_phi = optimum_cos;
		*sin_phi = optimum_sin;
		*distance = optimum_distance;
	}

	countOptimumTransformationOfMovedTriangleABCtoABCDEVICE(
		_ABC_A_x, _ABC_A_y, _ABC_B_x, _ABC_B_y, _ABC_C_x, _ABC_C_y,
		ABC_C_x, ABC_C_y, ABC_A_x, ABC_A_y, ABC_B_x, ABC_B_y,
		maxIterations, e, parts, step,
		&optimum_cos, &optimum_sin, &optimum_distance);

	if (optimum_distance < *distance)
	{
		*cos_phi = optimum_cos;
		*sin_phi = optimum_sin;
		*distance = optimum_distance;
	}
}


/*
	*ABC  - треугольники, к которым "стремимся"
	*_ABC - треугольники, которые будем крутить, вертеть
	в результате данной функции будет сформированы следующие матрицы:
	1) dx - массив переноса dx - центры масс треугольников ABC со знаком минус
	2) dy - аналогично
	3) distance, cos_phi, sin_phi - массивы размером _ABC_size * ABC_size

	тогда преобразование _ABCi -> ABCj:
	dx = dx[j]
	dy = dy[j]
	distance = distance[i * ABCsize + j]
	cos_phi = cos_phi[i * ABCsize + j]
	sin_phi = sin_phi[i * ABCsize + j]
*/
__global__ void kernel(
	float* _ABC_A_x, float* _ABC_A_y,
	float* _ABC_B_x, float* _ABC_B_y,
	float* _ABC_C_x, float* _ABC_C_y,
	int _ABC_size,
	float* ABC_A_x, float* ABC_A_y,
	float* ABC_B_x, float* ABC_B_y,
	float* ABC_C_x, float* ABC_C_y,
	int ABC_size,
	float* dx, float* dy,
	float* cos_phi, float* sin_phi, float* distance,
	int maxIterations, float e, int parts, float step)
{
	__shared__ float movedABC_A_x[16];
	__shared__ float movedABC_A_y[16];
	__shared__ float movedABC_B_x[16];
	__shared__ float movedABC_B_y[16];
	__shared__ float movedABC_C_x[16];
	__shared__ float movedABC_C_y[16];

	__shared__ float moved_ABC_A_x[16];
	__shared__ float moved_ABC_A_y[16];
	__shared__ float moved_ABC_B_x[16];
	__shared__ float moved_ABC_B_y[16];
	__shared__ float moved_ABC_C_x[16];
	__shared__ float moved_ABC_C_y[16];

	int row = defaultRow();
	int column = defaultColumn();

	if (threadIdx.y == 0 && column < ABC_size)
	{
		movedABC_A_x[threadIdx.x] = ABC_A_x[column];
		movedABC_B_x[threadIdx.x] = ABC_B_x[column];
		movedABC_C_x[threadIdx.x] = ABC_C_x[column];

		movedABC_A_y[threadIdx.x] = ABC_A_y[column];
		movedABC_B_y[threadIdx.x] = ABC_B_y[column];
		movedABC_C_y[threadIdx.x] = ABC_C_y[column];

		moveTriangleToStartDEVICE(
			&movedABC_A_x[threadIdx.x], &movedABC_A_y[threadIdx.x],
			&movedABC_B_x[threadIdx.x], &movedABC_B_y[threadIdx.x],
			&movedABC_C_x[threadIdx.x], &movedABC_C_y[threadIdx.x],
			&dx[column], &dy[column]);
	}
	__syncthreads();

	if (threadIdx.x == 0 && row < _ABC_size)
	{
		moved_ABC_A_x[threadIdx.y] = _ABC_A_x[row];
		moved_ABC_B_x[threadIdx.y] = _ABC_B_x[row];
		moved_ABC_C_x[threadIdx.y] = _ABC_C_x[row];

		moved_ABC_A_y[threadIdx.y] = _ABC_A_y[row];
		moved_ABC_B_y[threadIdx.y] = _ABC_B_y[row];
		moved_ABC_C_y[threadIdx.y] = _ABC_C_y[row];

		float tmp_dx; float tmp_dy;
		moveTriangleToStartDEVICE(
			&moved_ABC_A_x[threadIdx.y], &moved_ABC_A_y[threadIdx.y],
			&moved_ABC_B_x[threadIdx.y], &moved_ABC_B_y[threadIdx.y],
			&moved_ABC_C_x[threadIdx.y], &moved_ABC_C_y[threadIdx.y],
			&tmp_dx, &tmp_dy);
	}
	__syncthreads();

	if (row < _ABC_size && column < ABC_size)
	{
		int pos = row * ABC_size + column;
		countOptimumTransformationOfMovedTriangleDEVICE(
			moved_ABC_A_x[threadIdx.y], moved_ABC_A_y[threadIdx.y],
			moved_ABC_B_x[threadIdx.y], moved_ABC_B_y[threadIdx.y],
			moved_ABC_C_x[threadIdx.y], moved_ABC_C_y[threadIdx.y],
			movedABC_A_x[threadIdx.x], movedABC_A_y[threadIdx.x],
			movedABC_B_x[threadIdx.x], movedABC_B_y[threadIdx.x],
			movedABC_C_x[threadIdx.x], movedABC_C_y[threadIdx.x],
			maxIterations, e, parts, step,
			&cos_phi[pos], &sin_phi[pos], &distance[pos]);
	}
}

cudaError_t findOptimumTransformationWithoutStructuresCUDA(
	float* _ABC_A_x, float* _ABC_A_y,
	float* _ABC_B_x, float* _ABC_B_y,
	float* _ABC_C_x, float* _ABC_C_y,
	int _ABC_size,
	float* ABC_A_x, float* ABC_A_y,
	float* ABC_B_x, float* ABC_B_y,
	float* ABC_C_x, float* ABC_C_y,
	int ABC_size,
	float* dx, float* dy,
	float* cos_phi, float* sin_phi, float* distance,
	int maxIterations, float e, int parts, float step)
{
	float* dev_ABC_A_x; float* dev_ABC_A_y;
	float* dev_ABC_B_x; float* dev_ABC_B_y;
	float* dev_ABC_C_x; float* dev_ABC_C_y;
	
	float* devABC_A_x; float* devABC_A_y;
	float* devABC_B_x; float* devABC_B_y;
	float* devABC_C_x; float* devABC_C_y;

	float* devdx; float* devdy;
	float* devcos; float* devsin;
	float* devdistance;

	cudaError_t cudaStatus;
	
	//освобождаем память
	cudaStatus = cudaMalloc((void**)& dev_ABC_A_x, _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& dev_ABC_A_y, _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& dev_ABC_B_x, _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& dev_ABC_B_y, _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& dev_ABC_C_x, _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& dev_ABC_C_y, _ABC_size * sizeof(float)); checkStatus();

	cudaStatus = cudaMalloc((void**)& devABC_A_x, ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devABC_A_y, ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devABC_B_x, ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devABC_B_y, ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devABC_C_x, ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devABC_C_y, ABC_size * sizeof(float)); checkStatus();

	cudaStatus = cudaMalloc((void**)&devdx, ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)&devdy, ABC_size * sizeof(float)); checkStatus();
	
	cudaStatus = cudaMalloc((void**)&devcos, ABC_size * _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)&devsin, ABC_size * _ABC_size * sizeof(float)); checkStatus();
	cudaStatus = cudaMalloc((void**)&devdistance, ABC_size * _ABC_size * sizeof(float)); checkStatus();

	//копируем на видяху
	cudaStatus = cudaMemcpy(dev_ABC_A_x, _ABC_A_x, _ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(dev_ABC_B_x, _ABC_B_x, _ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(dev_ABC_C_x, _ABC_C_x, _ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(dev_ABC_A_y, _ABC_A_y, _ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(dev_ABC_B_y, _ABC_B_y, _ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(dev_ABC_C_y, _ABC_C_y, _ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();

	cudaStatus = cudaMemcpy(devABC_A_x, ABC_A_x, ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(devABC_B_x, ABC_B_x, ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(devABC_C_x, ABC_C_x, ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(devABC_A_y, ABC_A_y, ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(devABC_B_y, ABC_B_y, ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(devABC_C_y, ABC_C_y, ABC_size * sizeof(float), cudaMemcpyHostToDevice); checkStatus();

	//запускаем ядро
	dim3 threads(defaultThreadsCount, defaultThreadsCount);
	dim3 blocks(ceilMod(ABC_size, defaultThreadsCount), ceilMod(_ABC_size, defaultThreadsCount));
	kernel <<<blocks, threads>>>(
		dev_ABC_A_x, dev_ABC_A_y, dev_ABC_B_x, dev_ABC_B_y, dev_ABC_C_x, dev_ABC_C_y, _ABC_size,
		devABC_A_x, devABC_A_y, devABC_B_x, devABC_B_y, devABC_C_x, devABC_C_y, ABC_size,
		devdx, devdy, devcos, devsin, devdistance, maxIterations, e, parts, step);

	//проверяем на ошибки
	cudaStatus = cudaGetLastError(); checkStatus();
	cudaStatus = cudaDeviceSynchronize(); checkStatus();

	//копируем с видяхи
	cudaStatus = cudaMemcpy(dx, devdx, ABC_size * sizeof(float), cudaMemcpyDeviceToHost); checkStatus();
	cudaStatus = cudaMemcpy(dy, devdy, ABC_size * sizeof(float), cudaMemcpyDeviceToHost); checkStatus();

	cudaStatus = cudaMemcpy(cos_phi, devcos, ABC_size * _ABC_size * sizeof(float), cudaMemcpyDeviceToHost); checkStatus();
	cudaStatus = cudaMemcpy(sin_phi, devsin, ABC_size * _ABC_size * sizeof(float), cudaMemcpyDeviceToHost); checkStatus();
	cudaStatus = cudaMemcpy(distance, devdistance, ABC_size * _ABC_size * sizeof(float), cudaMemcpyDeviceToHost); checkStatus();

Error:
	cudaFree(dev_ABC_A_x);  cudaFree(dev_ABC_A_y);
	cudaFree(dev_ABC_B_x);  cudaFree(dev_ABC_B_y);
	cudaFree(dev_ABC_C_x);  cudaFree(dev_ABC_C_y);

	cudaFree(devABC_A_x);  cudaFree(devABC_A_y);
	cudaFree(devABC_B_x);  cudaFree(devABC_B_y);
	cudaFree(devABC_C_x);  cudaFree(devABC_C_y);

	cudaFree(devdx); cudaFree(devdy);
	cudaFree(devcos); cudaFree(devsin);
	cudaFree(devdistance);

	return cudaStatus;
}

int main()
{
	srand(time(NULL));
	const int maxRand = 100;
	const int maxIterations = 10;
	const int parts = 3;
	const float step = 2 * PI / parts;
	const float e = 0.00001f;

	int ABCsize = 200;
	float* ABC_A_x = (float*)malloc(ABCsize * sizeof(float));
	float* ABC_B_x = (float*)malloc(ABCsize * sizeof(float));
	float* ABC_C_x = (float*)malloc(ABCsize * sizeof(float));
	float* ABC_A_y = (float*)malloc(ABCsize * sizeof(float));
	float* ABC_B_y = (float*)malloc(ABCsize * sizeof(float));
	float* ABC_C_y = (float*)malloc(ABCsize * sizeof(float));
	for (int i = 0; i < ABCsize; i++)
	{
		ABC_A_x[i] = rand() % maxRand - maxRand / 2;
		ABC_B_x[i] = rand() % maxRand - maxRand / 2;
		ABC_C_x[i] = rand() % maxRand - maxRand / 2;
		
		ABC_A_y[i] = rand() % maxRand - maxRand / 2;
		ABC_B_y[i] = rand() % maxRand - maxRand / 2;
		ABC_C_y[i] = rand() % maxRand - maxRand / 2;
	}

	int _ABCsize = 100;
	float* _ABC_A_x = (float*)malloc(_ABCsize * sizeof(float));
	float* _ABC_B_x = (float*)malloc(_ABCsize * sizeof(float));
	float* _ABC_C_x = (float*)malloc(_ABCsize * sizeof(float));
	float* _ABC_A_y = (float*)malloc(_ABCsize * sizeof(float));
	float* _ABC_B_y = (float*)malloc(_ABCsize * sizeof(float));
	float* _ABC_C_y = (float*)malloc(_ABCsize * sizeof(float));
	for (int i = 0; i < _ABCsize; i++)
	{
		_ABC_A_x[i] = rand() % maxRand - maxRand / 2;
		_ABC_B_x[i] = rand() % maxRand - maxRand / 2;
		_ABC_C_x[i] = rand() % maxRand - maxRand / 2;

		_ABC_A_y[i] = rand() % maxRand - maxRand / 2;
		_ABC_B_y[i] = rand() % maxRand - maxRand / 2;
		_ABC_C_y[i] = rand() % maxRand - maxRand / 2;
	}

	float* dx = (float*)malloc(ABCsize * sizeof(float));
	float* dy = (float*)malloc(ABCsize * sizeof(float));
	float* cos_phi = (float*)malloc(ABCsize * _ABCsize * sizeof(float));
	float* sin_phi = (float*)malloc(ABCsize * _ABCsize * sizeof(float));
	float* distance = (float*)malloc(ABCsize * _ABCsize * sizeof(float));


    // Add vectors in parallel.
	
	cudaError_t cudaStatus = findOptimumTransformationWithoutStructuresCUDA(
		_ABC_A_x, _ABC_A_y, _ABC_B_x, _ABC_B_y, _ABC_C_x, _ABC_C_y, _ABCsize,
		ABC_A_x, ABC_A_y, ABC_B_x, ABC_B_y, ABC_C_x, ABC_C_y, ABCsize,
		dx, dy, cos_phi, sin_phi, distance,
		maxIterations, e, parts, step);

	checkStatus();
	cudaStatus = cudaDeviceReset(); checkStatus();
    return 0;
Error:
	return 1;
}
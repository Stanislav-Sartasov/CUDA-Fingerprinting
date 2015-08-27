#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define defaultRow() blockIdx.y*blockDim.y + threadIdx.y
#define defaultColumn() blockIdx.x*blockDim.x + threadIdx.x
#define ceilMod(x ,y) (x + y - 1)/(y)

#define checkStatus() if (cudaStatus != cudaSuccess) goto Error

#define PI 3.141592f
const int defaultThreadCount = 16;

typedef struct { float x; float y; } Point;
typedef struct { Point A; Point B; Point C; } Triangle;
typedef struct { float dx; float dy; float sin_phi; float cos_phi; } Transformation;
typedef struct { Transformation transformation; float distance; } TransformationWithDistance;

//точки
__host__ float countDistanceBetweenPointsHOST(Point first, Point second)
{
	return (first.x - second.x)*(first.x - second.x) + (first.y - second.y)*(first.y - second.y);
}
__host__ Point countMovedPointHOST(Point p, float dx, float dy)
{
	Point result;
	result.x = p.x + dx;
	result.y = p.y + dy;
	return result;
}
__host__ Point countRotatedPointHOST(Point p, float cos_phi, float sin_phi)
{
	Point result;
	result.x = p.x * cos_phi - p.y * sin_phi;
	result.y = p.x * sin_phi + p.y * cos_phi;
	return result;
}

//треугольники
__host__ float countDistanceBetweenTrianglesHOST(Triangle* first, Triangle* second)
{
	return
		countDistanceBetweenPointsHOST(first->A, second->A) +
		countDistanceBetweenPointsHOST(first->B, second->B) +
		countDistanceBetweenPointsHOST(first->C, second->C);
}
__host__ Point countTriangleMassCenterHOST(Triangle* ABC)
{
	Point result;
	result.x = (ABC->A.x + ABC->B.x + ABC->C.x) / 3;
	result.y = (ABC->A.y + ABC->B.y + ABC->C.y) / 3;
	return result;
}
__host__ Triangle countMovedTriangleHOST(Triangle* ABC, float dx, float dy)
{
	Triangle result;
	result.A = countMovedPointHOST(ABC->A, dx, dy);
	result.B = countMovedPointHOST(ABC->B, dx, dy);
	result.C = countMovedPointHOST(ABC->C, dx, dy);
	return result;
}
__host__ Triangle countRotatedTriangleHOST(Triangle* ABC, float cos_phi, float sin_phi)
{
	Triangle result;
	result.A = countRotatedPointHOST(ABC->A, cos_phi, sin_phi);
	result.B = countRotatedPointHOST(ABC->B, cos_phi, sin_phi);
	result.C = countRotatedPointHOST(ABC->C, cos_phi, sin_phi);
	return result;
}
__host__ Triangle countTransformedTriangleHOST(Triangle* ABC, Transformation t)
{
	Point ABCmc = countTriangleMassCenterHOST(ABC);
	Triangle ABC_moved = countMovedTriangleHOST(ABC, -ABCmc.x, -ABCmc.y);
	Triangle ABC_moved_rotated = countRotatedTriangleHOST(&ABC_moved, t.cos_phi, t.sin_phi);
	return countMovedTriangleHOST(&ABC_moved_rotated, t.dx, t.dy);
}

//преобразовани€
__host__ float countOptimumlPhiHOST(float phi0, float sProd, float vProd, int maxIterations, float e)
{
	float resultPhi = phi0;

	float cos_phi;
	float sin_phi;
	float newPhi;
	for (int i = 0; i < maxIterations; i++)
	{
		cos_phi = cosf(resultPhi);
		sin_phi = sinf(resultPhi);

		newPhi = resultPhi - (sin_phi * sProd + cos_phi * vProd) / (cos_phi * sProd - sin_phi * vProd);

		if (fabsf(newPhi - resultPhi) < e)
			return newPhi;
		else
			resultPhi = newPhi;
	}

	return resultPhi;
}
__host__ TransformationWithDistance findOptimumTransformationABCHOST(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts)
{
	Point ABCmc = countTriangleMassCenterHOST(ABC);
	Triangle movedABC = countMovedTriangleHOST(ABC, -ABCmc.x, -ABCmc.y);

	Point ABC_mc = countTriangleMassCenterHOST(ABC_);
	Triangle movedABC_ = countMovedTriangleHOST(ABC_, -ABC_mc.x, -ABC_mc.y);

	float sProd = movedABC.A.x * movedABC_.A.x + movedABC.A.y * movedABC_.A.y + movedABC.B.x * movedABC_.B.x + movedABC.B.y * movedABC_.B.y + movedABC.C.x * movedABC_.C.x + movedABC.C.y * movedABC_.C.y;
	float vProd = movedABC.A.x * movedABC_.A.y - movedABC.A.y * movedABC_.A.x + movedABC.B.x * movedABC_.B.y - movedABC.B.y * movedABC_.B.x + movedABC.C.x * movedABC_.C.y - movedABC.C.y * movedABC_.C.x;

	TransformationWithDistance result;
	result.transformation.dx = ABCmc.x; result.transformation.dy = ABCmc.y;
	result.transformation.cos_phi = 1; result.transformation.sin_phi = 0;
	result.distance = countDistanceBetweenTrianglesHOST(&movedABC, &movedABC_);

	float optimumPhi;
	float optimum_cos;
	float optimum_sin;
	float step = 2 * PI / parts;
	Triangle tmpResult;
	float distance;

	for (int i = 0; i < parts; i++)
	{
		optimumPhi = countOptimumlPhiHOST(i * step - PI, sProd, vProd, maxIterations, e);
		optimum_cos = cosf(optimumPhi);
		optimum_sin = sinf(optimumPhi);

		tmpResult = countRotatedTriangleHOST(&movedABC_, optimum_cos, optimum_sin);

		distance = countDistanceBetweenTrianglesHOST(&movedABC, &tmpResult);
		if (distance < result.distance)
		{
			result.distance = distance;
			result.transformation.cos_phi = optimum_cos;
			result.transformation.sin_phi = optimum_sin;
		}
	}

	return result;
}
__host__ TransformationWithDistance findOptimumTransformationHOST(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts)
{
	TransformationWithDistance twdABC = findOptimumTransformationABCHOST(ABC_, ABC, e, maxIterations, parts);

	Triangle tmpTriangle;
	tmpTriangle.A = ABC_->B;
	tmpTriangle.B = ABC_->C;
	tmpTriangle.C = ABC_->A;
	TransformationWithDistance twdBCA = findOptimumTransformationABCHOST(&tmpTriangle, ABC, e, maxIterations, parts);

	tmpTriangle.A = ABC_->C;
	tmpTriangle.B = ABC_->A;
	tmpTriangle.C = ABC_->B;
	TransformationWithDistance twdCAB = findOptimumTransformationABCHOST(&tmpTriangle, ABC, e, maxIterations, parts);

	if (twdBCA.distance < twdABC.distance)
		twdABC = twdBCA;

	return (twdCAB.distance < twdABC.distance) ? twdCAB : twdABC;
}


//точки

/*
¬ычисление рассто€ни€ между двум€ точками, без извлечени€ корн€.
*/
__device__ float countDistanceBetweenPointsDEVICE(Point first, Point second)
{
	return (first.x - second.x)*(first.x - second.x) + (first.y - second.y)*(first.y - second.y);
}
/*
¬ычисление образа точки, сдвинутой относительно исходной на dx, dy
*/
__device__ Point countMovedPointDEVICE(Point p, float dx, float dy)
{
	Point result;
	result.x = p.x + dx;
	result.y = p.y + dy;
	return result;
}
/*
¬ычисление образа точки, после поворота относительно начала координат на угол phi, 
нужно передать его синус и косинус
*/
__device__ Point countRotatedPointDEVICE(Point p, float cos_phi, float sin_phi)
{
	Point result;
	result.x = p.x * cos_phi - p.y * sin_phi;
	result.y = p.x * sin_phi + p.y * cos_phi;
	return result;
}


//треугольники

/*
¬ычисление суммы рассто€ний между соответсвующими вершинами двух треугольников
*/
__device__ float countDistanceBetweenTrianglesDEVICE(Triangle* first, Triangle* second)
{
	return
		countDistanceBetweenPointsDEVICE(first->A, second->A) +
		countDistanceBetweenPointsDEVICE(first->B, second->B) +
		countDistanceBetweenPointsDEVICE(first->C, second->C);
}
/*
¬ычисление центра масс
*/
__device__ Point countTriangleMassCenterDEVICE(Triangle* ABC)
{
	Point result;
	result.x = (ABC->A.x + ABC->B.x + ABC->C.x) / 3;
	result.y = (ABC->A.y + ABC->B.y + ABC->C.y) / 3;
	return result;
}
/*
¬ычисление образа треугольника, передвинутого на dx, dy
*/
__device__ Triangle countMovedTriangleDEVICE(Triangle* ABC, float dx, float dy)
{
	Triangle result;
	result.A = countMovedPointDEVICE(ABC->A, dx, dy);
	result.B = countMovedPointDEVICE(ABC->B, dx, dy);
	result.C = countMovedPointDEVICE(ABC->C, dx, dy);
	return result;
}
/*
¬ычисление образа треугольника, повернутого относительно начала координат на угол phi
необходимо передать его синус и косинус
*/
__device__ Triangle countRotatedTriangleDEVICE(Triangle* ABC, float cos_phi, float sin_phi)
{
	Triangle result;
	result.A = countRotatedPointDEVICE(ABC->A, cos_phi, sin_phi);
	result.B = countRotatedPointDEVICE(ABC->B, cos_phi, sin_phi);
	result.C = countRotatedPointDEVICE(ABC->C, cos_phi, sin_phi);
	return result;
}
/*
	1)сдвигаем треугольник в начало координат
	2)поворачиваем
	3)передвигаем на dx, dy
*/
__device__ Triangle countTransformedTriangleDEVICE(Triangle* ABC, Transformation t)
{
	Point ABCmc = countTriangleMassCenterDEVICE(ABC);
	Triangle ABC_moved = countMovedTriangleDEVICE(ABC, -ABCmc.x, -ABCmc.y);
	Triangle ABC_moved_rotated = countRotatedTriangleDEVICE(&ABC_moved, t.cos_phi, t.sin_phi);
	return countMovedTriangleDEVICE(&ABC_moved_rotated, t.dx, t.dy);
}


//преобразовани€

/*F = (Ax - A_x cos_phi + A_y sin_phi)^2 + (Ay - A_x sin_phi - A_y cos_phi)^2 + 
(Bx - B_x cos_phi + B_y sin_phi)^2 + (By - B_x sin_phi - B_y cos_phi)^2 +
(Cx - C_x cos_phi + C_y sin_phi)^2 + (Cy - C_x sin_phi - C_y cos_phi)^2;  - еЄ нужно минимализировать
ƒл€ этого: вычисл€ем дифференциал, второй дифференциал а после - находим нули производной с:
phi0 - начальное приближение, maxIterations - максимальным числом итераций и допустимой погрешностью e*/
__device__ float countOptimumlPhiDEVICE(float phi0, float sProd, float vProd, int maxIterations, float e)
{
	float resultPhi = phi0;

	float cos_phi;
	float sin_phi;
	float newPhi;
	for (int i = 0; i < maxIterations; i++)
	{
		cos_phi = cosf(resultPhi);
		sin_phi = sinf(resultPhi);

		newPhi = resultPhi - (sin_phi * sProd + cos_phi * vProd) / (cos_phi * sProd - sin_phi * vProd);

		if (fabsf(newPhi - resultPhi) < e)
			return newPhi;
		else
			resultPhi = newPhi;
	}

	return resultPhi;
}
/*
	Ќаходим оптимальное преобразование ABC_ -> ABC, начальные приближени€ - (360/parts - 180) * i
*/
__device__ TransformationWithDistance findOptimumTransformationABCDEVICE(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts)
{
	Point ABCmc = countTriangleMassCenterDEVICE(ABC);
	Triangle movedABC = countMovedTriangleDEVICE(ABC, -ABCmc.x, -ABCmc.y);

	Point ABC_mc = countTriangleMassCenterDEVICE(ABC_);
	Triangle movedABC_ = countMovedTriangleDEVICE(ABC_, -ABC_mc.x, -ABC_mc.y);

	float sProd = movedABC.A.x * movedABC_.A.x + movedABC.A.y * movedABC_.A.y + movedABC.B.x * movedABC_.B.x + movedABC.B.y * movedABC_.B.y + movedABC.C.x * movedABC_.C.x + movedABC.C.y * movedABC_.C.y;
	float vProd = movedABC.A.x * movedABC_.A.y - movedABC.A.y * movedABC_.A.x + movedABC.B.x * movedABC_.B.y - movedABC.B.y * movedABC_.B.x + movedABC.C.x * movedABC_.C.y - movedABC.C.y * movedABC_.C.x;

	TransformationWithDistance result;
	result.transformation.dx = ABCmc.x; result.transformation.dy = ABCmc.y;
	result.transformation.cos_phi = 1; result.transformation.sin_phi = 0;
	result.distance = countDistanceBetweenTrianglesDEVICE(&movedABC, &movedABC_);

	float optimumPhi;
	float optimum_cos;
	float optimum_sin;
	float step = 2 * PI / parts;
	Triangle tmpResult;
	float distance;

	for (int i = 0; i < parts; i++)
	{
		optimumPhi = countOptimumlPhiDEVICE(i * step - PI, sProd, vProd, maxIterations, e);
		optimum_cos = cosf(optimumPhi);
		optimum_sin = sinf(optimumPhi);

		tmpResult = countRotatedTriangleDEVICE(&movedABC_, optimum_cos, optimum_sin);

		distance = countDistanceBetweenTrianglesDEVICE(&movedABC, &tmpResult);
		if (distance < result.distance)
		{
			result.distance = distance;
			result.transformation.cos_phi = optimum_cos;
			result.transformation.sin_phi = optimum_sin;
		}
	}

	return result;
}
/*
Ќаходим оптимальное преобазование среди ABC_ -> ABC, BCA_ -> ABC, CBA_ -> ABC 
*/
__device__ TransformationWithDistance findOptimumTransformationDEVICE(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts)
{
	TransformationWithDistance twdABC = findOptimumTransformationABCDEVICE(ABC_, ABC, e, maxIterations, parts);

	Triangle tmpTriangle;
	tmpTriangle.A = ABC_->B;
	tmpTriangle.B = ABC_->C;
	tmpTriangle.C = ABC_->A;
	TransformationWithDistance twdBCA = findOptimumTransformationABCDEVICE(&tmpTriangle, ABC, e, maxIterations, parts);

	tmpTriangle.A = ABC_->C;
	tmpTriangle.B = ABC_->A;
	tmpTriangle.C = ABC_->B;
	TransformationWithDistance twdCAB = findOptimumTransformationABCDEVICE(&tmpTriangle, ABC, e, maxIterations, parts);

	if (twdBCA.distance < twdABC.distance)
		twdABC = twdBCA;

	return (twdCAB.distance < twdABC.distance) ? twdCAB : twdABC;
}


typedef enum{ TYPE_ABC, TYPE_BCA, TYPE_CAB }TriangleType;
__host__ TransformationWithDistance findOptimumTransformationABCParallelForHOST(Triangle* ABC_, TriangleType ABC_type, Triangle* ABC, float e, int maxIterations, int step, float stepSize)
{
	Triangle newABC_;
	if (ABC_type == TYPE_BCA)
	{
		newABC_.A = ABC_->B;
		newABC_.B = ABC_->C;
		newABC_.C = ABC_->A;
	}
	else
		if (ABC_type == TYPE_CAB)
		{
			newABC_.A = ABC_->C;
			newABC_.B = ABC_->A;
			newABC_.C = ABC_->B;
		}
		else
			newABC_ = *ABC_;

	Point ABC_mc = countTriangleMassCenterHOST(&newABC_);
	Triangle movedABC_ = countMovedTriangleHOST(&newABC_, -ABC_mc.x, -ABC_mc.y);

	Point ABCmc = countTriangleMassCenterHOST(ABC);
	Triangle movedABC = countMovedTriangleHOST(ABC, -ABCmc.x, -ABCmc.y);

	float sProd = movedABC.A.x * movedABC_.A.x + movedABC.A.y * movedABC_.A.y + movedABC.B.x * movedABC_.B.x + movedABC.B.y * movedABC_.B.y + movedABC.C.x * movedABC_.C.x + movedABC.C.y * movedABC_.C.y;
	float vProd = movedABC.A.x * movedABC_.A.y - movedABC.A.y * movedABC_.A.x + movedABC.B.x * movedABC_.B.y - movedABC.B.y * movedABC_.B.x + movedABC.C.x * movedABC_.C.y - movedABC.C.y * movedABC_.C.x;

	TransformationWithDistance result;
	result.transformation.dx = ABCmc.x; result.transformation.dy = ABCmc.y;
	result.transformation.cos_phi = 1; result.transformation.sin_phi = 0;
	result.distance = countDistanceBetweenTrianglesHOST(&movedABC, &movedABC_);

	float optimumPhi = countOptimumlPhiHOST(step * stepSize - PI, sProd, vProd, maxIterations, e);
	float optimum_cos = cosf(optimumPhi);
	float optimum_sin = sinf(optimumPhi);

	Triangle tmpTriangle = countRotatedTriangleHOST(&movedABC_, optimum_cos, optimum_sin);
	float distance = countDistanceBetweenTrianglesHOST(&movedABC, &tmpTriangle);
	if (distance < result.distance)
	{
		result.distance = distance;
		result.transformation.cos_phi = optimum_cos;
		result.transformation.sin_phi = optimum_sin;
	}

	return result;
}
__host__ TransformationWithDistance findOptimumTransformationParallelForHOST(Triangle* ABC_, Triangle* ABC, int maxIterations, float e, int parts)
{
	TransformationWithDistance* transformations = (TransformationWithDistance*)malloc(parts * 3 * sizeof(TransformationWithDistance));
	
	float stepSize = 2 * PI / parts;

	for (int i = 0; i < parts; i++)
	{
		transformations[i] = findOptimumTransformationABCParallelForHOST(ABC_, TYPE_ABC, ABC, e, maxIterations, i, stepSize);
		transformations[i + parts] = findOptimumTransformationABCParallelForHOST(ABC_, TYPE_BCA, ABC, e, maxIterations, i, stepSize);
		transformations[i + 2 * parts] = findOptimumTransformationABCParallelForHOST(ABC_, TYPE_CAB, ABC, e, maxIterations, i, stepSize);
	}
	
	TransformationWithDistance result = transformations[0];
	for (int i = 0; i < 3 * parts; i++)
		if (transformations[i].distance < result.distance)
			result = transformations[i];

	free(transformations);
	return result;
}

__device__ TransformationWithDistance findOptimumTransformationABCParallelForDEVICE(Triangle* ABC_, TriangleType ABC_type, Triangle* ABC, float e, int maxIterations, int step, float stepSize)
{
	Triangle newABC_;
	if (ABC_type == TYPE_BCA)
	{
		newABC_.A = ABC_->B;
		newABC_.B = ABC_->C;
		newABC_.C = ABC_->A;
	}
	else
		if (ABC_type == TYPE_CAB)
		{
			newABC_.A = ABC_->C;
			newABC_.B = ABC_->A;
			newABC_.C = ABC_->B;
		}
		else
			newABC_ = *ABC_;

	Point ABC_mc = countTriangleMassCenterDEVICE(&newABC_);
	Triangle movedABC_ = countMovedTriangleDEVICE(&newABC_, -ABC_mc.x, -ABC_mc.y);

	Point ABCmc = countTriangleMassCenterDEVICE(ABC);
	Triangle movedABC = countMovedTriangleDEVICE(ABC, -ABCmc.x, -ABCmc.y);

	float sProd = movedABC.A.x * movedABC_.A.x + movedABC.A.y * movedABC_.A.y + movedABC.B.x * movedABC_.B.x + movedABC.B.y * movedABC_.B.y + movedABC.C.x * movedABC_.C.x + movedABC.C.y * movedABC_.C.y;
	float vProd = movedABC.A.x * movedABC_.A.y - movedABC.A.y * movedABC_.A.x + movedABC.B.x * movedABC_.B.y - movedABC.B.y * movedABC_.B.x + movedABC.C.x * movedABC_.C.y - movedABC.C.y * movedABC_.C.x;

	TransformationWithDistance result;
	result.transformation.dx = ABCmc.x; result.transformation.dy = ABCmc.y;
	result.transformation.cos_phi = 1; result.transformation.sin_phi = 0;
	result.distance = countDistanceBetweenTrianglesDEVICE(&movedABC, &movedABC_);

	float optimumPhi = countOptimumlPhiDEVICE(step * stepSize - PI, sProd, vProd, maxIterations, e);
	float optimum_cos = cosf(optimumPhi);
	float optimum_sin = sinf(optimumPhi);

	Triangle tmpTriangle = countRotatedTriangleDEVICE(&movedABC_, optimum_cos, optimum_sin);
	float distance = countDistanceBetweenTrianglesDEVICE(&movedABC, &tmpTriangle);
	if (distance < result.distance)
	{
		result.distance = distance;
		result.transformation.cos_phi = optimum_cos;
		result.transformation.sin_phi = optimum_sin;
	}

	return result;
}

//cuda
//parallel for version
__global__ void findOptimumTransormationParallelForVersion(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, float e, int maxIterations, int parts)
{
	extern __shared__ TransformationWithDistance cacheTWD[];
	__shared__ Triangle ABCcache;
	__shared__ float stepSize;

	/*
    7
   ABC_
	y
	^
	|
	--> x ABC 200


	ABC_ -> row    -> y
	ABC  -> column -> x
	#define defaultRow() blockIdx.y*blockDim.y + threadIdx.y
	#define defaultColumn() blockIdx.x*blockDim.x + threadIdx.x
	*/

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		ABCcache = ABC[blockIdx.x];
		stepSize = 2 * PI / parts;
	}
	
	__syncthreads();

	
	int row = defaultRow();

	if (row < ABC_size && threadIdx.x < 3 * parts)
	{
		Triangle ABC_cache = ABC_[row];
		int pos = threadIdx.x + threadIdx.y * 3 * parts;
		cacheTWD[pos] = findOptimumTransformationABCParallelForDEVICE(&ABC_cache, TriangleType(threadIdx.x % 3), &ABCcache, e, maxIterations, threadIdx.x % parts, stepSize);
	}

	__syncthreads();
	if (threadIdx.x == 0)
	{
		TransformationWithDistance resultTWD = cacheTWD[threadIdx.y * 3 * parts];
		for (int i = 1; i < parts * 3; i++)
		{
			if (cacheTWD[threadIdx.y * 3 * parts + i].distance < resultTWD.distance)
				resultTWD = cacheTWD[threadIdx.y * 3 * parts + i];
		}
		
		result[row * ABCsize + blockIdx.x] = resultTWD;
	}
}
cudaError_t findOptimumTransformationParallelForWithCuda(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts)
{

	Triangle* devABC_;
	Triangle* devABC;
	TransformationWithDistance* devResult;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)& devABC_, ABC_size * sizeof(Triangle));
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void**)& devABC, ABCsize * sizeof(Triangle));
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void**)& devResult, ABC_size * ABCsize * sizeof(TransformationWithDistance));
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(devABC_, ABC_, ABC_size * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(devABC, ABC, ABCsize * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	//findOptimumTransormationParallelForVersion(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, float e, int maxIterations, int parts)
	dim3 threads(defaultThreadCount, defaultThreadCount);
	dim3 blocks(ABCsize, ceilMod(ABC_size, defaultThreadCount));
	findOptimumTransormationParallelForVersion <<< blocks, threads, defaultThreadCount * defaultThreadCount * sizeof(TransformationWithDistance) >>>(devABC_, ABC_size, devABC, ABCsize, devResult, e, maxIterations, parts);


	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(result, devResult, ABC_size * ABCsize * sizeof(TransformationWithDistance), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

Error:
	cudaFree(devABC_);
	cudaFree(devABC);
	cudaFree(devResult);

	return cudaStatus;
}

//non-parellel for version

//non-parellel for version
__global__ void fOTKernel(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts)
{
	//extern __shared__ Triangle cache[];
	int row = defaultRow();
	int column = defaultColumn();

	if (row < ABC_size && column < ABCsize)
	{
		Triangle abc_ = ABC_[row];
		Triangle abc = ABC[column];
		result[row * ABCsize + column] = findOptimumTransformationDEVICE(&abc_, &abc, e, maxIterations, parts);
	}
}
cudaError_t findOptimumTransformationNonParallelForWithCuda(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts)
{
	Triangle* devABC_;
	Triangle* devABC;
	TransformationWithDistance* devResult;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)& devABC_, ABC_size * sizeof(Triangle));
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void**)& devABC, ABCsize * sizeof(Triangle));
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMalloc((void**)& devResult, ABC_size * ABCsize * sizeof(TransformationWithDistance));
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(devABC_, ABC_, ABC_size * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(devABC, ABC, ABCsize * sizeof(Triangle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		goto Error;

	dim3 threads(defaultThreadCount, defaultThreadCount);
	dim3 blocks(ceilMod(ABC_size, defaultThreadCount), ceilMod(ABCsize, defaultThreadCount));

	//void findOptimumTransformationKernel(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts)
	fOTKernel <<< blocks, threads >>>(devABC_, ABC_size, devABC, ABCsize, devResult, maxIterations, e, parts);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		goto Error;

	cudaStatus = cudaMemcpy(result, devResult, ABC_size * ABCsize * sizeof(TransformationWithDistance), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		goto Error;

Error:
	cudaFree(devABC_);
	cudaFree(devABC);
	cudaFree(devResult);

	return cudaStatus;
}


//transformation + distance version
__global__ void findOptimumTransformationPlusDistanceVersionKernel(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, Transformation* result, float* resultDistance, int maxIterations, float e, int parts)
{
	//extern __shared__ Triangle cache[];
	int row = defaultRow();
	int column = defaultColumn();

	if (row < ABC_size && column < ABCsize)
	{
		Triangle abc_ = ABC_[row];
		Triangle abc = ABC[column];
		TransformationWithDistance toResult = findOptimumTransformationDEVICE(&abc_, &abc, e, maxIterations, parts);
		result[row * ABCsize + column] = toResult.transformation;
		resultDistance[row * ABCsize + column] = toResult.distance;
	}
}
cudaError_t findOptimumTransformationPlusDistanceVersionCUDA(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, Transformation* result, float* resultDistance, int maxIterations, float e, int parts)
{
	Triangle* devABC_;
	Triangle* devABC;
	Transformation* devResult;
	float* devDistance;

	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)& devABC_, ABC_size * sizeof(Triangle)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devABC, ABCsize * sizeof(Triangle)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devResult, ABC_size * ABCsize * sizeof(Transformation)); checkStatus();
	cudaStatus = cudaMalloc((void**)& devDistance, ABC_size * ABCsize * sizeof(float)); checkStatus();

	cudaStatus = cudaMemcpy(devABC_, ABC_, ABC_size * sizeof(Triangle), cudaMemcpyHostToDevice); checkStatus();
	cudaStatus = cudaMemcpy(devABC, ABC, ABCsize * sizeof(Triangle), cudaMemcpyHostToDevice); checkStatus();

	dim3 threads(defaultThreadCount, defaultThreadCount);
	dim3 blocks(ceilMod(ABC_size, defaultThreadCount), ceilMod(ABCsize, defaultThreadCount));

	//void findOptimumTransformationKernel(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts)
	findOptimumTransformationPlusDistanceVersionKernel << < blocks, threads >> >(devABC_, ABC_size, devABC, ABCsize, devResult, devDistance, maxIterations, e, parts);

	cudaStatus = cudaGetLastError(); checkStatus();
	cudaStatus = cudaDeviceSynchronize(); checkStatus();

	cudaStatus = cudaMemcpy(result, devResult, ABC_size * ABCsize * sizeof(Transformation), cudaMemcpyDeviceToHost); checkStatus();
	cudaStatus = cudaMemcpy(resultDistance, devDistance, ABC_size * ABCsize * sizeof(float), cudaMemcpyDeviceToHost); checkStatus();

Error:
	cudaFree(devABC_);
	cudaFree(devABC);
	cudaFree(devResult);
	cudaFree(devDistance);

	return cudaStatus;
}


int main()
{
	int max_rand = 100;

	int ABC_size = 100;
	int ABCsize = 200;
	Triangle* ABC = (Triangle*)malloc(ABCsize * sizeof(Triangle));
	Triangle* ABC_ = (Triangle*)malloc(ABC_size * sizeof(Triangle));

	srand(time(NULL));                      // инициализаци€ функции rand значением функции time

	for (int i = 0; i < ABCsize; i++)
	{
		Triangle ABCt;

		ABCt.A.x = max_rand / 2 - rand() % max_rand;
		ABCt.A.y = max_rand / 2 - rand() % max_rand;

		ABCt.B.x = max_rand / 2 - rand() % max_rand;
		ABCt.B.y = max_rand / 2 - rand() % max_rand;

		ABCt.C.x = max_rand / 2 - rand() % max_rand;
		ABCt.C.y = max_rand / 2 - rand() % max_rand;
		ABC[i] = ABCt;
	}

	for (int i = 0; i < ABC_size; i++)
	{
		Triangle ABCt;

		ABCt.A.x = max_rand / 2 - rand() % max_rand;
		ABCt.A.y = max_rand / 2 - rand() % max_rand;

		ABCt.B.x = max_rand / 2 - rand() % max_rand;
		ABCt.B.y = max_rand / 2 - rand() % max_rand;

		ABCt.C.x = max_rand / 2 - rand() % max_rand;
		ABCt.C.y = max_rand / 2 - rand() % max_rand;
		ABC_[i] = ABCt;
	}


	TransformationWithDistance* result = (TransformationWithDistance*)malloc(ABC_size * ABCsize * sizeof(TransformationWithDistance));
	
	Transformation* resultTransformation = (Transformation*)malloc(ABC_size * ABCsize * sizeof(Transformation));
	float* resultDistance = (float*)malloc(ABC_size * ABCsize * sizeof(float));
	cudaError_t cudaStatus = findOptimumTransformationPlusDistanceVersionCUDA(ABC_, ABC_size, ABC, ABCsize, resultTransformation, resultDistance, 10, 0.00001f, 5);
	
	Transformation zeroResult = resultTransformation[56 * 200 + 4];
	float zeroDistance = resultDistance[56 * 200 + 4];

	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = findOptimumTransformationParallelForWithCuda(ABC_, ABC_size, ABC, ABCsize, result, 10, 0.00001f, 5);
	TransformationWithDistance firstResult = result[56 * 200 + 4];
	
	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = findOptimumTransformationNonParallelForWithCuda(ABC_, ABC_size, ABC, ABCsize, result, 10, 0.00001f, 5);
	TransformationWithDistance secondResult = result[56 * 200 + 4];
	
	if (cudaStatus != cudaSuccess)
		goto End;
		
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		goto End;

	TransformationWithDistance dest = findOptimumTransformationHOST(&ABC_[56], &ABC[4], 0.00001f, 10, 5);
	TransformationWithDistance dest2 = findOptimumTransformationParallelForHOST(&ABC_[56], &ABC[4], 10, 0.00001f, 5);
	float distance = countDistanceBetweenTrianglesHOST(&ABC_[56], &ABC[4]);

End:
	free(ABC);
	free(ABC_);
	free(result);

	return 0;
}
#include "TrianglesMatcherWithCUDA.cuh"
#include <math.h>

#define defaultRow() blockIdx.y*blockDim.y + threadIdx.y
#define defaultColumn() blockIdx.x*blockDim.x + threadIdx.x
#define ceilMod(x ,y) (x + y - 1)/(y)
#define checkStatus() if (cudaStatus != cudaSuccess) goto Error

#ifndef PI
#define PI 3.141592f
#endif

const int defaultThreadCount = 16;

//точки

/*
¬ычисление рассто€ни€ между двум€ точками, без извлечени€ корн€.
*/
__host__ __device__ float countDistanceBetweenPoints(Point first, Point second)
{
	return (first.x - second.x)*(first.x - second.x) + (first.y - second.y)*(first.y - second.y);
}
/*
¬ычисление образа точки, сдвинутой относительно исходной на dx, dy
*/
__host__ __device__ Point countMovedPoint(Point p, float dx, float dy)
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
__host__ __device__ Point countRotatedPoint(Point p, float cos_phi, float sin_phi)
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
__host__ __device__ float countDistanceBetweenTriangles(Triangle* first, Triangle* second)
{
	return
		countDistanceBetweenPoints(first->A, second->A) +
		countDistanceBetweenPoints(first->B, second->B) +
		countDistanceBetweenPoints(first->C, second->C);
}
/*
¬ычисление центра масс
*/
__host__ __device__ Point countTriangleMassCenter(Triangle* ABC)
{
	Point result;
	result.x = (ABC->A.x + ABC->B.x + ABC->C.x) / 3;
	result.y = (ABC->A.y + ABC->B.y + ABC->C.y) / 3;
	return result;
}
/*
¬ычисление образа треугольника, передвинутого на dx, dy
*/
__host__ __device__ Triangle countMovedTriangle(Triangle* ABC, float dx, float dy)
{
	Triangle result;
	result.A = countMovedPoint(ABC->A, dx, dy);
	result.B = countMovedPoint(ABC->B, dx, dy);
	result.C = countMovedPoint(ABC->C, dx, dy);
	return result;
}
/*
¬ычисление образа треугольника, повернутого относительно начала координат на угол phi
необходимо передать его синус и косинус
*/
__host__ __device__ Triangle countRotatedTriangle(Triangle* ABC, float cos_phi, float sin_phi)
{
	Triangle result;
	result.A = countRotatedPoint(ABC->A, cos_phi, sin_phi);
	result.B = countRotatedPoint(ABC->B, cos_phi, sin_phi);
	result.C = countRotatedPoint(ABC->C, cos_phi, sin_phi);
	return result;
}
/*
1)сдвигаем треугольник в начало координат
2)поворачиваем
3)передвигаем на dx, dy
*/
__host__ __device__ Triangle countTransformedTriangle(Triangle* ABC, Transformation t)
{
	Point ABCmc = countTriangleMassCenter(ABC);
	Triangle ABC_moved = countMovedTriangle(ABC, -ABCmc.x, -ABCmc.y);
	Triangle ABC_moved_rotated = countRotatedTriangle(&ABC_moved, t.cos_phi, t.sin_phi);
	return countMovedTriangle(&ABC_moved_rotated, t.dx, t.dy);
}


//преобразовани€

/*
phi0 - начальное приближение, maxIterations - максимальным числом итераций и допустимой погрешностью e*/
__host__ __device__ float countOptimumlPhi(float phi0, float sProd, float vProd, int maxIterations, float e)
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
__host__ __device__ TransformationWithDistance findOptimumTransformationABC(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts)
{
	Point ABCmc = countTriangleMassCenter(ABC);
	Triangle movedABC = countMovedTriangle(ABC, -ABCmc.x, -ABCmc.y);

	Point ABC_mc = countTriangleMassCenter(ABC_);
	Triangle movedABC_ = countMovedTriangle(ABC_, -ABC_mc.x, -ABC_mc.y);

	float sProd = movedABC.A.x * movedABC_.A.x + movedABC.A.y * movedABC_.A.y + movedABC.B.x * movedABC_.B.x + movedABC.B.y * movedABC_.B.y + movedABC.C.x * movedABC_.C.x + movedABC.C.y * movedABC_.C.y;
	float vProd = movedABC.A.x * movedABC_.A.y - movedABC.A.y * movedABC_.A.x + movedABC.B.x * movedABC_.B.y - movedABC.B.y * movedABC_.B.x + movedABC.C.x * movedABC_.C.y - movedABC.C.y * movedABC_.C.x;

	TransformationWithDistance result;
	result.transformation.dx = ABCmc.x; result.transformation.dy = ABCmc.y;
	result.transformation.cos_phi = 1; result.transformation.sin_phi = 0;
	result.distance = countDistanceBetweenTriangles(&movedABC, &movedABC_);

	float optimumPhi;
	float optimum_cos;
	float optimum_sin;
	float step = 2 * PI / parts;
	Triangle tmpResult;
	float distance;

	for (int i = 0; i < parts; i++)
	{
		optimumPhi = countOptimumlPhi(i * step - PI, sProd, vProd, maxIterations, e);
		optimum_cos = cosf(optimumPhi);
		optimum_sin = sinf(optimumPhi);

		tmpResult = countRotatedTriangle(&movedABC_, optimum_cos, optimum_sin);

		distance = countDistanceBetweenTriangles(&movedABC, &tmpResult);
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
__host__ __device__ TransformationWithDistance findOptimumTransformation(Triangle* ABC_, Triangle* ABC, float e, int maxIterations, int parts)
{
	TransformationWithDistance twdABC = findOptimumTransformationABC(ABC_, ABC, e, maxIterations, parts);

	Triangle tmpTriangle;
	tmpTriangle.A = ABC_->B;
	tmpTriangle.B = ABC_->C;
	tmpTriangle.C = ABC_->A;
	TransformationWithDistance twdBCA = findOptimumTransformationABC(&tmpTriangle, ABC, e, maxIterations, parts);

	tmpTriangle.A = ABC_->C;
	tmpTriangle.B = ABC_->A;
	tmpTriangle.C = ABC_->B;
	TransformationWithDistance twdCAB = findOptimumTransformationABC(&tmpTriangle, ABC, e, maxIterations, parts);

	if (twdBCA.distance < twdABC.distance)
		twdABC = twdBCA;

	return (twdCAB.distance < twdABC.distance) ? twdCAB : twdABC;
}



//cuda

//non-parellel for version
__global__ void findOptimumTransformationNonParallelForKernel(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, int maxIterations, float e, int parts)
{
	//extern __shared__ Triangle cache[];
	int row = defaultRow();
	int column = defaultColumn();

	if (row < ABC_size && column < ABCsize)
	{
		Triangle abc_ = ABC_[row];
		Triangle abc = ABC[column];
		result[row * ABCsize + column] = findOptimumTransformation(&abc_, &abc, e, maxIterations, parts);
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
	findOptimumTransformationNonParallelForKernel << < blocks, threads >> >(devABC_, ABC_size, devABC, ABCsize, devResult, maxIterations, e, parts);

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

//parallel for version
__device__ TransformationWithDistance findOptimumTransformationABCParallelFor(Triangle* ABC_, TriangleType ABC_type, Triangle* ABC, float e, int maxIterations, int step, float stepSize)
{
	//мен€ем пор€док вершин, если надо
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

	Point ABC_mc = countTriangleMassCenter(&newABC_);
	Triangle movedABC_ = countMovedTriangle(&newABC_, -ABC_mc.x, -ABC_mc.y);

	Point ABCmc = countTriangleMassCenter(ABC);
	Triangle movedABC = countMovedTriangle(ABC, -ABCmc.x, -ABCmc.y);

	float sProd = movedABC.A.x * movedABC_.A.x + movedABC.A.y * movedABC_.A.y + movedABC.B.x * movedABC_.B.x + movedABC.B.y * movedABC_.B.y + movedABC.C.x * movedABC_.C.x + movedABC.C.y * movedABC_.C.y;
	float vProd = movedABC.A.x * movedABC_.A.y - movedABC.A.y * movedABC_.A.x + movedABC.B.x * movedABC_.B.y - movedABC.B.y * movedABC_.B.x + movedABC.C.x * movedABC_.C.y - movedABC.C.y * movedABC_.C.x;

	TransformationWithDistance result;
	result.transformation.dx = ABCmc.x; result.transformation.dy = ABCmc.y;
	result.transformation.cos_phi = 1; result.transformation.sin_phi = 0;
	result.distance = countDistanceBetweenTriangles(&movedABC, &movedABC_);

	float optimumPhi = countOptimumlPhi(step * stepSize - PI, sProd, vProd, maxIterations, e);
	float optimum_cos = cosf(optimumPhi);
	float optimum_sin = sinf(optimumPhi);

	Triangle tmpTriangle = countRotatedTriangle(&movedABC_, optimum_cos, optimum_sin);
	float distance = countDistanceBetweenTriangles(&movedABC, &tmpTriangle);
	if (distance < result.distance)
	{
		result.distance = distance;
		result.transformation.cos_phi = optimum_cos;
		result.transformation.sin_phi = optimum_sin;
	}

	return result;
}
__global__ void findOptimumTransormationParallelForKernel(Triangle* ABC_, int ABC_size, Triangle* ABC, int ABCsize, TransformationWithDistance* result, float e, int maxIterations, int parts)
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
		cacheTWD[pos] = findOptimumTransformationABCParallelFor(&ABC_cache, TriangleType(threadIdx.x % 3), &ABCcache, e, maxIterations, threadIdx.x % parts, stepSize);
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
	findOptimumTransormationParallelForKernel << < blocks, threads, defaultThreadCount * defaultThreadCount * sizeof(TransformationWithDistance) >> >(devABC_, ABC_size, devABC, ABCsize, devResult, e, maxIterations, parts);


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
		TransformationWithDistance toResult = findOptimumTransformation(&abc_, &abc, e, maxIterations, parts);
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

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BinTemplateCorrelation.cu"
#include "CylinderHelper.cuh"
#include "constsmacros.h"
#include "ConvexHull.cu"
#include "CUDAArray.cuh"
#include <stdio.h>
#include "math_constants.h"
#include "VectorHelper.cuh"
#include "math.h"
#include "ConvexHull.cu"
#include "ConvexHullModified.cu"
#include "math.h"
struct Minutia
{
	float angle;
	int x;
	int y;
};

struct Consts
{
	const char radius = 70;
	const char baseCuboid = 8;
	const char heightCuboid = 5;
	const unsigned int numberCell = baseCuboid * baseCuboid * heightCuboid;
	const float baseCell = (2.0 * radius) / baseCuboid;
	const float heightCell = (2 * CUDART_PI) / heightCuboid;
	const float sigmaLocation = 28.0 / 3;
	const float sigmaDirection = 2 * CUDART_PI / 9;
	const float sigmoidParametrPsi = 0.01;
	const char omega = 50;
	const char minNumberMinutiae = 2;
};

__constant__ Consts constsGPU;

__device__ __host__ Point getPoint(int i, int j, Minutia minutia)
{
	return Point(
		(float)
		(minutia.x + constsGPU.baseCell *
		(cos(minutia.angle) * (i - (constsGPU.baseCuboid + 1) / 2.0) +
		sin(minutia.angle) * (j - (constsGPU.baseCuboid + 1) / 2.0))),
		(float)
		(minutia.y + constsGPU.baseCell *
		(-sin(minutia.angle) * (i - (constsGPU.baseCuboid + 1) / 2.0) +
		cos(minutia.angle) * (j - (constsGPU.baseCuboid + 1) / 2.0)))
		);
}

__device__ __host__ CUDAArray<Minutia*> getNeighborhood(Point point, Minutia middleMinutia, CUDAArray<Minutia> minutiaArr)
{
	int count = 0;
	CUDAArray<Minutia*> tmp = CUDAArray<Minutia*>(minutiaArr.Width, minutiaArr.Height);
	for (size_t i; i < minutiaArr.Height*minutiaArr.Width; i++)
	{
		if ((pointDistance(Point((float)minutiaArr.At(0, i).x, (float)(minutiaArr.At(0, i).y)), point) < 3 * constsGPU.sigmaLocation) &&
			(!equalsMinutae(minutiaArr.At(0, i), middleMinutia)))
		{
			tmp.SetAt(0, count, &minutiaArr.At(0, i));
			count++;
		}
	}
	CUDAArray<Minutia*> neighborhood = CUDAArray<Minutia*>(0, count);
	for (size_t i = 0; i < count; i++)
	{
		neighborhood.SetAt(0, i, tmp.At(0, i));
	}
	tmp.Dispose();
	return neighborhood;
}

__device__ __host__ float angleHeight(int k)
{
	return (-CUDART_PI + (k - 0.5) * constsGPU.heightCell);
}

__device__ __host__ float gaussian1D(float x, float sigma)
{
	float commonDenom = 2 * sigma * sigma;
	float denominator = sigma * sqrtf(CUDART_PI * 2);
	float result = expf(-(x * x) / commonDenom) / denominator;
	return result;
}

__device__ __host__ float gaussianLocation(Minutia minutia, Point point)
{
	return gaussian1D(pointDistance(Point(minutia.x, minutia.y), point),
		constsGPU.sigmaLocation);
}
__device__ __host__ float gaussianDirection(Minutia middleMinutia, Minutia minutia, float anglePoint)
{
	float common = sqrt(2.0) * constsGPU.sigmaDirection;
	double angle = getAngleDiff(anglePoint,
		getAngleDiff(middleMinutia.angle, minutia.angle));
	double first = erf(((angle + constsGPU.heightCell / 2)) / common);
	double second = erf(((angle - constsGPU.heightCell / 2)) / common);
	return (first - second) / 2;
}

__inline__ __device__ __host__ bool equalsMinutae(Minutia firstMinutia, Minutia secondMinutia)
{
	return (
		firstMinutia.x == secondMinutia.x &&
		firstMinutia.y == secondMinutia.y &&
		abs(firstMinutia.angle - secondMinutia.angle) < 1.401298E-45
		);
}

__device__ __host__ bool isValidPoint(Point point, Minutia middleMinutia, Point* hull, int hullLength)
{
	return pointDistance(Point(middleMinutia.x, middleMinutia.y), point) < constsGPU.radius &&
		isPointInsideHull(point, hull, hullLength);
}

__device__ __host__ float sum(Point point, double anglePoint, CUDAArray<Minutia*> neighborhood, Minutia middleMinutia)
{
	double sum = 0;
	for (size_t i = 0; i < neighborhood.Width * neighborhood.Height; i++)
	{
		sum += gaussianLocation((*neighborhood.At(0, i)), point) * gaussianDirection(middleMinutia, *neighborhood.At(0, i), anglePoint);
	}
	return sum;
}

__device__ __host__ char stepFunction(float value)
{
	return (char)(value >= constsGPU.sigmoidParametrPsi ? 1 : 0);
}

void createTemplate(Minutia* minutiae, int lenght, Cylinder* cylinders, int* cylindersLenght)
{
	Consts myConst;
	cudaMemcpyToSymbol(&constsGPU, &myConst, sizeof(Consts));
	Point* points = (Point*)malloc(lenght * sizeof(Point));
	CUDAArray<Minutia> cudaMinutiae = CUDAArray<Minutia>(minutiae, lenght, 1);
	CUDAArray<Point> cudaPoints = CUDAArray<Point>(points, lenght, 1);
	free(points);
	getPoints << <1, lenght >> >(cudaMinutiae, cudaPoints, lenght);
	int* hullLenght;
	Point* hull;

	getConvexHull(cudaPoints.GetData, lenght, hull, hullLenght);
	cudaPoints.Dispose();
	Point* exdHull = extendHull(hull, *hullLenght, 50);//50 - omega 
	free(hull);
	int* extLenght;
	*extLenght = *hullLenght * 2;
	free(hullLenght);
	bool* isValidMinutiae = (bool*)malloc(lenght*sizeof(bool));
	CUDAArray<bool> cudaIsValidMinutiae = CUDAArray<bool>(isValidMinutiae, lenght, 1);
	getValidMinutias << <1, lenght >> >(cudaMinutiae, isValidMinutiae);
	cudaMinutiae.Dispose();
	cudaIsValidMinutiae.GetData(isValidMinutiae);
	cudaIsValidMinutiae.Dispose();
	int count = 0;
	for (int i = 0; i < lenght; i++)
	{
		if (isValidMinutiae[i])
		{
			count++;
		}
	}
	count = 0;
	Minutia* validMinutiae = (Minutia*)malloc(count*sizeof(Minutia));
	for (int i = 0; i < lenght; i++)
	{
		if (isValidMinutiae[i])
		{
			validMinutiae[count] = minutiae[i];
			count++;
		}
	}//count - validMinutiaeLenght	
	free(isValidMinutiae);

}

__global__ void createCylinders(CUDAArray<Minutia> minutiae, CUDAArray<CylinderGPU> cylinders, Point* hull, int hullLenght)
{
	if (threadIdx.x + 1>16 || threadIdx.y + 1 > 16 || threadIdx.z + 1 > 3 || blockIdx.x > minutiae.Width || blockIdx.y > 2)
	{
		return;
	}
	if (isValidPoint(getPoint(threadIdx.x, threadIdx.y, minutiae.At(0, blockIdx.x)), minutiae.At(0, blockIdx.x), hull, hullLenght))
	{
		
	}
}

__global__ void getValidMinutias(CUDAArray<Minutia> minutiae, CUDAArray<bool> isValidMinutiae)
{
	if (threadIdx.x < minutiae.Width)
	{
		int count = 0;
		for (int i = 0; i < minutiae.Width; i++)
		{
			if (threadIdx.x != i)
			{
				count = sqrt((float)
					((minutiae.At(0, threadIdx.x).x - minutiae.At(0, i).x)*(minutiae.At(0, threadIdx.x).x - minutiae.At(0, i).x) +
					minutiae.At(0, threadIdx.x).y - minutiae.At(0, i).y)*(minutiae.At(0, threadIdx.x).y - minutiae.At(0, i).y))
					< constsGPU.radius + 3 * constsGPU.sigmaLocation ? count + 1 : count;
			}
		}
		isValidMinutiae.SetAt(0, threadIdx.x, count >= 2 ? true : false);
	}
}

__global__ void getPoints(CUDAArray<Minutia> minutiae, CUDAArray<Point> points, int lenght)
{
	if (threadIdx.x < lenght)
	{
		points.SetAt(0, threadIdx.x, Point(minutiae.At(0, threadIdx.x).x, minutiae.At(0, threadIdx.x).y));
	}
}

int main()
{

}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "BinTemplateCorrelation.cu"
#include "CylinderHelper.cuh"
#include "ConvexHull.cu"
#include <stdio.h>
#include "math_constants.h"
#include "VectorHelper.cuh"
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

__device__ __host__ Point getPoint(int i, int j, Minutia minutia, float baseCell, float baseCuboid)
{
	return Point(
		(float)
			(minutia.x + baseCell *
			(cos(minutia.angle) * (i - (baseCuboid + 1) / 2.0) +
			sin(minutia.angle) * (j - (baseCuboid + 1) / 2.0))),
		(float)
			(minutia.y + baseCell *
			(-sin(minutia.angle) * (i - (baseCuboid + 1) / 2.0) +
			cos(minutia.angle) * (j - (baseCuboid + 1) / 2.0)))
		);
}

__device__ __host__ CUDAArray<Minutia*> getNeighborhood(Point point, Minutia middleMinutia, CUDAArray<Minutia> minutiaArr, float sigmaLocation)
{
	int count = 0;
	CUDAArray<Minutia*> tmp = CUDAArray<Minutia*>(minutiaArr.Width, minutiaArr.Height);
	for (size_t i; i < minutiaArr.Height*minutiaArr.Width; i++)
	{
		if ((pointDistance(Point((float)minutiaArr.At(0, i).x, (float)(minutiaArr.At(0, i).y)), point) < 3 * sigmaLocation) &&
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

__device__ __host__ float angleHeight(int k, float heightCell)
{
	return (- CUDART_PI + (k - 0.5) * heightCell);
}

__device__ __host__ float gaussian1D(float x, float sigma)
{
	float commonDenom = 2 * sigma * sigma;
	float denominator = sigma * sqrtf(CUDART_PI * 2);
	float result = expf(-(x * x) / commonDenom) / denominator;
	return result;
}

__device__ __host__ float gaussianLocation(Minutia minutia, Point point, float sigmaLocation)
{
	return gaussian1D(pointDistance(Point(minutia.x, minutia.y), point),
		sigmaLocation);
}
__device__ __host__ float gaussianDirection(Minutia middleMinutia, Minutia minutia, float anglePoint, float sigmaDirection, float heightCell)
{
	float common = sqrt(2.0) * sigmaDirection;
	double angle = getAngleDiff(anglePoint,
		getAngleDiff(middleMinutia.angle, minutia.angle));
	double first = erf(((angle + heightCell / 2)) / common);
	double second = erf(((angle - heightCell / 2)) / common);
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

__device__ __host__ bool isValidPoint(Point point, Minutia middleMinutia, char radius, Point* hull, int hullLength)
{
	return pointDistance(Point(middleMinutia.x, middleMinutia.y), point) < radius &&
		isPointInsideHull(point, hull, hullLength);
}

__device__ __host__ float sum(Point point, double anglePoint, CUDAArray<Minutia*> neighborhood, Minutia middleMinutia, float sigmaLocation, float sigmaDirection, float heightCell)
{
	double sum = 0;
	for (size_t i = 0; i < neighborhood.Width * neighborhood.Height; i++)
	{
		sum += gaussianLocation((*neighborhood.At(0, i)), point, sigmaLocation) * gaussianDirection(middleMinutia, *neighborhood.At(0, i), anglePoint, sigmaDirection, heightCell);
	}
	return sum;
}

__device__ __host__ char stepFunction(float value, float sigmoidParametrPsi)
{
	return (char)(value >= sigmoidParametrPsi ? 1 : 0);
}
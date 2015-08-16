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
#include "CreateTemplate.h"
#include "device_functions_decls.h"

__device__  Point* getPoint(Minutia *minutiae)
{
	return &Point(
		(float)
		((*minutiae).x + constsGPU.baseCell *
		(cos((*minutiae).angle) * (defaultX() - (constsGPU.baseCuboid + 1) / 2.0) +
		sin((*minutiae).angle) * (defaultY() - (constsGPU.baseCuboid + 1) / 2.0))),
		(float)
		((*minutiae).y + constsGPU.baseCell *
		(-sin((*minutiae).angle) * (defaultX() - (constsGPU.baseCuboid + 1) / 2.0) +
		cos((*minutiae).angle) * (defaultY() - (constsGPU.baseCuboid + 1) / 2.0)))
		);
}

__device__ CUDAArray<Minutia*> getNeighborhood(CUDAArray<Minutia> *minutiaArr)
{
	int count = 0;
	CUDAArray<Minutia*> tmp = CUDAArray<Minutia*>((*minutiaArr).Width, (*minutiaArr).Height);
	for (size_t i = 0; i < (*minutiaArr).Height*(*minutiaArr).Width; i++)
	{
		if ((pointDistance(Point((float)(*minutiaArr).At(0, i).x, (float)((*minutiaArr).At(0, i).y)), *getPoint(&(*minutiaArr).At(0, defaultMinutia())))) < 3 * constsGPU.sigmaLocation &&
			(!equalsMinutae((*minutiaArr).AtPtr(0, i), (*minutiaArr).AtPtr(0, defaultMinutia()))))
		{
			tmp.SetAt(0, count, &(*minutiaArr).At(0, i));
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

__device__  float angleHeight()
{
	return (-CUDART_PI + (defaultZ() - 0.5) * constsGPU.heightCell);
}

__device__ __host__ float gaussian1D(float x)
{
	return expf(-(x * x) / (2 * constsGPU.sigmaLocation * constsGPU.sigmaLocation)) / (constsGPU.sigmaLocation * sqrtf(CUDART_PI * 2));
}

__device__ __host__ float gaussianLocation(Minutia *minutia, Point *point)
{
	return gaussian1D(pointDistance(Point((*minutia).x, (*minutia).y), *point));
}

__device__ float gaussianDirection(Minutia *middleMinutia, Minutia *minutia, float anglePoint)
{
	float common = sqrt(2.0) * constsGPU.sigmaDirection;
	double angle = getAngleDiff(anglePoint,
		getAngleDiff((*middleMinutia).angle, (*minutia).angle));
	double first = erf(((angle + constsGPU.heightCell / 2)) / common);
	double second = erf(((angle - constsGPU.heightCell / 2)) / common);
	return (first - second) / 2;
}

__inline__ __device__ bool equalsMinutae(Minutia* firstMinutia, Minutia* secondMinutia)
{
	return (
		(*firstMinutia).x == (*secondMinutia).x &&
		(*firstMinutia).y == (*secondMinutia).y &&
		abs((*firstMinutia).angle - (*secondMinutia).angle) < 1.401298E-45
		);
}

__device__ __host__ bool isValidPoint(Minutia* middleMinutia, Point* hull, int hullLength)
{
	return pointDistance(Point((*middleMinutia).x, (*middleMinutia).y), *getPoint(middleMinutia)) < constsGPU.radius &&
		isPointInsideHull(*getPoint(middleMinutia), hull, hullLength);
}

__device__ __host__ float sum(CUDAArray<Minutia*> neighborhood, Minutia* middleMinutia)
{
	double sum = 0;
	for (size_t i = 0; i < neighborhood.Width * neighborhood.Height; i++)
	{
		sum += gaussianLocation(&(*neighborhood.At(0, i)), getPoint(middleMinutia)) * gaussianDirection(middleMinutia, neighborhood.At(0, i), angleHeight());
	}
	return sum;
}

__device__ __host__ char stepFunction(float value)
{
	return (char)(value >= constsGPU.sigmoidParametrPsi ? 1 : 0);
}

void createTemplate(Minutia* minutiae, int lenght, Cylinder* cylinders, int* cylindersLenght)
{
	cudaSetDevice(0);
	Consts myConst;
	cudaMemcpyToSymbol(&constsGPU, &myConst, sizeof(Consts));
	Point* points = (Point*)malloc(lenght * sizeof(Point));
	CUDAArray<Minutia> cudaMinutiae = CUDAArray<Minutia>(minutiae, lenght, 1);
	CUDAArray<Point> cudaPoints = CUDAArray<Point>(points, lenght, 1);
	free(points);
	getPoints << <1, lenght >> >(cudaMinutiae, cudaPoints, lenght);
	int* hullLenght;
	Point* hull = (Point*)malloc(lenght*sizeof(Point));

	getConvexHull(cudaPoints.GetData(), lenght, hull, hullLenght);
	cudaPoints.Dispose();
	Point* extHull = extendHull(hull, *hullLenght, constsGPU.omega);
	free(hull);
	int* extLenght;
	*extLenght = *hullLenght * 2;
	free(hullLenght);
	cudaMalloc((void**)&hullGPU, sizeof(Point)*(*extLenght));
	cudaMemcpy(hullGPU, extHull, sizeof(Point)*(*extLenght), cudaMemcpyHostToDevice);
	free(extHull);
	bool* isValidMinutiae = (bool*)malloc(lenght*sizeof(bool));
	CUDAArray<bool> cudaIsValidMinutiae = CUDAArray<bool>(isValidMinutiae, lenght, 1);
	getValidMinutias << <1, lenght >> >(cudaMinutiae, cudaIsValidMinutiae);
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
	unsigned int** valuesAndMasks = (unsigned int**)malloc(count*sizeof(unsigned int*));
	for (int i = 0; i < count; i++)
	{
		valuesAndMasks[i] = (unsigned int*)malloc(2 * 48 * sizeof(unsigned int));
	}
	CUDAArray <unsigned int> cudaValuesAndMasks = CUDAArray<unsigned int>(*valuesAndMasks, 96, count);
	createValuesAndMasks << < dim3(count, 2), 48 >> >(cudaMinutiae, cudaValuesAndMasks, hullLenght)
		/*Cylinder value =
		int maxNorm = 0;
		for (int i = 0;i<)*/

}

__global__ void createValuesAndMasks(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> valuesAndMasks, int hullLenght)
{
	if (defaultX() > 16 || defaultY() > 16 || defaultZ() > 6 || defaultMinutia() > minutiae.Width)
	{
		return;
	}
	if (isValidPoint(&minutiae.At(0, defaultMinutia()), hull, hullLenght))
	{
		char tempValue =
			(defaultY() % 2)*(stepFunction(sum(getNeighborhood(&minutiae), &(minutiae.At(0, defaultMinutia())))));
		atomicOr(valuesAndMasks.AtPtr(defaultMinutia(), curIndex()), (tempValue - '0' + blockIdx.y) << linearizationIndex() % 32);
	}
	else
	{
		atomicOr(valuesAndMasks.AtPtr(defaultMinutia(), curIndex()), 0 << linearizationIndex() % 32);
	}
}

__global__ void IsValidCylinders(CUDAArray<Cylinder> masksCylinders, CUDAArray<bool> isValidCylinders)
{

}

__global__ void createCylinders(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> sum, CUDAArray<unsigned int> values, CUDAArray<unsigned int> masks, CUDAArray<Cylinder> cylinders)
{
	cylinders.SetAt(0, blockIdx.x * 2, Cylinder(values.AtPtr(blockIdx.x, 0), values.Width,
		minutiae.At(0, blockIdx.x).angle, sqrt((float)(sum.At(0, blockIdx.x))), 0));
	cylinders.SetAt(0, blockIdx.x * 2 + 1, Cylinder(masks.AtPtr(blockIdx.x, 0), masks.Width,
		minutiae.At(0, blockIdx.x).angle, sqrt((float)(sum.At(1, blockIdx.x))), 0));
}

__global__ void createNorm(CUDAArray<unsigned int> valuesAndMasks, CUDAArray<unsigned int> sum)
{
	if (blockIdx.y == 0)
	{
		unsigned int x = __popc(valuesAndMasks.At(defaultMinutia(), threadIdx.x * 2 + blockIdx.y));
		atomicAdd(sum.AtPtr(blockIdx.y, blockIdx.x), x);
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
		isValidMinutiae.SetAt(0, threadIdx.x, count >= constsGPU.minNumberMinutiae ? true : false);
	}
}

__global__ void getPoints(CUDAArray<Minutia> minutiae, CUDAArray<Point> points, int lenght)
{
	if (threadIdx.x < lenght)
	{
		points.SetAt(0, threadIdx.x, Point(minutiae.At(0, threadIdx.x).x, minutiae.At(0, threadIdx.x).y));
	}
}


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constsmacros.h"
#include "BinTemplateCorrelation.cu"
#include "CylinderHelper.cuh"
#include "ConvexHull.cuh"
#include "CUDAArray.cuh"
#include "math_constants.h"
#include "VectorHelper.cuh"
#include "math.h"
#include "math.h"
#include "TemplateCreation.cuh"
#include "device_functions_decls.h"
#include "ConvexHullModified.cuh"
#include <stdio.h>

__device__  Point* getPoint(Minutia *minutiae)
{
	return &Point(
		(float)
		((*minutiae).x + (*constsGPU).baseCell *
		(cos((*minutiae).angle) * (defaultX() - ((*constsGPU).baseCuboid + 1) / 2.0) +
		sin((*minutiae).angle) * (defaultY() - ((*constsGPU).baseCuboid + 1) / 2.0))),
		(float)
		((*minutiae).y + (*constsGPU).baseCell *
		(-sin((*minutiae).angle) * (defaultX() - ((*constsGPU).baseCuboid + 1) / 2.0) +
		cos((*minutiae).angle) * (defaultY() - ((*constsGPU).baseCuboid + 1) / 2.0)))
		);
}

/*__device__ void getNeighborhood(CUDAArray<Minutia> *minutiaArr, Minutia** neighborhood, int* lenghtNeighborhood)
{
	int validMinutiaeLenght = 0;
	Minutia* tmp[200];
	for (size_t i = 0; i < (*minutiaArr).Height*(*minutiaArr).Width; i++)
	{
		if ((pointDistance(Point((float)(*minutiaArr).At(0, i).x, (float)((*minutiaArr).At(0, i).y)), 
			*getPoint(&(*minutiaArr).At(0, defaultMinutia())))) < 3 * (*constsGPU).sigmaLocation &&
			(!equalsMinutae((*minutiaArr).AtPtr(0, i), (*minutiaArr).AtPtr(0, defaultMinutia()))))
		{
			tmp[validMinutiaeLenght] = &((*minutiaArr).At(0, i));
			validMinutiaeLenght++;
		}
	}
	for (size_t i = 0; i < validMinutiaeLenght; i++)
	{
		neighborhood[i] = tmp[i];
		(*lenghtNeighborhood)++;
	}
}*/


__device__  float angleHeight()
{
	return (-CUDART_PI + (defaultZ() - 0.5) * (*constsGPU).heightCell);
}

__device__  float gaussian1D(float x)
{
	return expf(-(x * x) / (2 * (*constsGPU).sigmaLocation * (*constsGPU).sigmaLocation)) / ((*constsGPU).sigmaLocation * sqrtf(CUDART_PI * 2));
}
/*not work
__device__  float gaussianLocation(Minutia *minutia, Point *point)
{
	return gaussian1D(pointDistance(Point((*minutia).x, (*minutia).y), *point));
}
*/

__device__ float gaussianDirection(Minutia *middleMinutia, Minutia *minutia, float anglePoint)
{
	float common = sqrt(2.0) * (*constsGPU).sigmaDirection;
	double angle = getAngleDiff(anglePoint,
		getAngleDiff((*middleMinutia).angle, (*minutia).angle));
	double first = erf(((angle + (*constsGPU).heightCell / 2)) / common);
	double second = erf(((angle - (*constsGPU).heightCell / 2)) / common);
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
/*not work
__device__ bool isValidPoint(Minutia* middleMinutia)
{
	return pointDistance(Point((*middleMinutia).x, (*middleMinutia).y), *getPoint(middleMinutia)) < (*constsGPU).radius &&
		isPointInsideHull(*getPoint(middleMinutia), hullGPU, *hullLenghtGPU);
}
*/
/*not work
__device__ float sum(Minutia** neighborhood, Minutia* middleMinutia, int lenghtNeigborhood)
{
	double sum = 0;
	for (size_t i = 0; i < lenghtNeigborhood; i++)
	{
		sum += gaussianLocation(&(*neighborhood[i]), getPoint(middleMinutia)) * gaussianDirection(middleMinutia, neighborhood[i], angleHeight());
	}
	return sum;
}
*/

__device__ char stepFunction(float value)
{
	return (char)(value >= (*constsGPU).sigmoidParametrPsi ? 1 : 0);
}

__global__ void getPoints(CUDAArray<Minutia> minutiae, CUDAArray<Point> points)
{
	if (threadIdx.x < minutiae.Width)
	{
		points.SetAt(0, threadIdx.x, Point(minutiae.At(0, threadIdx.x).x, minutiae.At(0, threadIdx.x).y));
	}
}

__global__ void getValidMinutiae(CUDAArray<Minutia> minutiae, CUDAArray<bool> isValidMinutiae)
{
	if (threadIdx.x >= minutiae.Width)
	{
		return;
	}
	int validMinutiaeLenght = 0;
	for (int i = 0; i < minutiae.Width; i++)
	{
		if (threadIdx.x == i)
		{
			continue;
		}
		validMinutiaeLenght = sqrt((float)
			((minutiae.At(0, threadIdx.x).x - minutiae.At(0, i).x)*(minutiae.At(0, threadIdx.x).x - minutiae.At(0, i).x) +
			minutiae.At(0, threadIdx.x).y - minutiae.At(0, i).y)*(minutiae.At(0, threadIdx.x).y - minutiae.At(0, i).y))
			< (*constsGPU).radius + 3 * (*constsGPU).sigmaLocation ? validMinutiaeLenght + 1 : validMinutiaeLenght;
	}
	isValidMinutiae.SetAt(0, threadIdx.x, validMinutiaeLenght >= (*constsGPU).minNumberMinutiae ? true : false);
}

__global__ void createSum(CUDAArray<unsigned int> valuesAndMasks, CUDAArray<unsigned int> sum)
{
	unsigned int x = __popc(valuesAndMasks.At(defaultMinutia(), threadIdx.x * 2 + blockIdx.x));
	atomicAdd(sum.AtPtr(0, threadIdx.x * 2 + blockIdx.x), x);
}


__global__ void createCylinders(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> sum, 
	CUDAArray<unsigned int> valuesAndMasks, CUDAArray<Cylinder> cylinders)
{
	cylinders.SetAt(0, blockIdx.x, Cylinder(valuesAndMasks.AtPtr(blockIdx.x, 0), valuesAndMasks.Width,
		minutiae.At(0, blockIdx.x).angle, sqrt((float)(sum.At(0, blockIdx.x))), 0));
}
/*not work
__global__ void createValuesAndMasks(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> valuesAndMasks, Minutia ** neighborhood)
{
	int lenghtNeighborhood = 0;
	if (defaultX() > 16 || defaultY() > 16 || defaultZ() > 6 || defaultMinutia() > minutiae.Width)
	{
		return;
	}
	if (isValidPoint(&minutiae.At(0, defaultMinutia())))
	{
		char tempValue =
			(defaultY() % 2)*(stepFunction(sum(getNeighborhood(&minutiae, &lenghtNeighborhood), &(minutiae.At(0, defaultMinutia())), lenghtNeighborhood)));
		atomicOr(valuesAndMasks.AtPtr(defaultMinutia(), curIndex()), (tempValue - '0' + blockIdx.y) << linearizationIndex() % 32);
	}
	else
	{
		atomicOr(valuesAndMasks.AtPtr(defaultMinutia(), curIndex()), 0 << linearizationIndex() % 32);
	}
}
*/
int main()
{
   
}

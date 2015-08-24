#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constsmacros.h"
#include "TemplateCreation.cuh"
#include "BinTemplateCorrelation.cu"
#include "ConvexHull.cu"
#include "CUDAArray.cuh"
#include "math_constants.h"
#include "VectorHelper.cu"
#include "math.h"
#include "device_functions_decls.h"
#include "ConvexHullModified.cu"
#include "CylinderHelper.cuh"
#include <stdio.h>

__device__  Point* getPoint(Minutia *minutiae, int i, int j)
{
	return &Point(
		(float)
		((*minutiae).x + constsGPU[0].baseCell *
		(cos((*minutiae).angle) * (i - (constsGPU[0].baseCuboid + 1) / 2.0f) +
		sin((*minutiae).angle) * (j - (constsGPU[0].baseCuboid + 1) / 2.0f))),
		(float)
		((*minutiae).y + constsGPU[0].baseCell *
		(-sin((*minutiae).angle) * (i - (constsGPU[0].baseCuboid + 1) / 2.0f) +
		cos((*minutiae).angle) * (j - (constsGPU[0].baseCuboid + 1) / 2.0f)))
		);
}

__device__  float angleHeight()
{
	return (-CUDART_PI + (defaultZ() - 0.5) * constsGPU[0].heightCell);
}

__device__  float gaussian1D(float x)
{
	return expf(-(x * x) / (2 * constsGPU[0].sigmaLocation * constsGPU[0].sigmaLocation)) / (constsGPU[0].sigmaLocation * sqrtf(CUDART_PI * 2));
}

__device__ float getPointDistance(Point A, Point B)
{
	float diffX = B.x - A.x;
	float diffY = B.y - A.y;

	return sqrt(diffX * diffX + diffY * diffY);
}

__device__ float gaussianLocation(Minutia *minutia, Point *point)
{
	return gaussian1D(getPointDistance(Point((*minutia).x, (*minutia).y), *point));
}

__device__ float gaussianDirection(Minutia *middleMinutia, Minutia *minutia, float anglePoint)
{
	float common = sqrt(2.0) * constsGPU[0].sigmaDirection;
	float angle = getAngleDiff(anglePoint,
		getAngleDiff((*middleMinutia).angle, (*minutia).angle));
	float first = erf(((angle + constsGPU[0].heightCell / 2)) / common);
	float second = erf(((angle - constsGPU[0].heightCell / 2)) / common);
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

__device__ bool isValidPoint(Minutia* middleMinutia)
{
	return  getPointDistance(Point((*middleMinutia).x, (*middleMinutia).y), *getPoint(middleMinutia, defaultX(), defaultY())) < constsGPU[0].radius &&
		isPointInsideHull(*getPoint(middleMinutia, defaultX(), defaultY()), hullGPU, hullLenghtGPU[0]);
}

__device__ char stepFunction(float value)
{
	return (char)(value >= constsGPU[0].sigmoidParametrPsi ? 1 : 0);
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
		validMinutiaeLenght =
			sqrt((float)((minutiae.At(0, threadIdx.x).x - minutiae.At(0, i).x)*
			(minutiae.At(0, threadIdx.x).x - minutiae.At(0, i).x) +
			(minutiae.At(0, threadIdx.x).y - minutiae.At(0, i).y)*
			(minutiae.At(0, threadIdx.x).y - minutiae.At(0, i).y))) ?
			validMinutiaeLenght + 1 : validMinutiaeLenght;
	}
	isValidMinutiae.SetAt(0, threadIdx.x, validMinutiaeLenght >= constsGPU[0].minNumberMinutiae ? true : false);
}

__global__ void createSum(CUDAArray<unsigned int> valuesAndMasks, unsigned int* sum)
{
	unsigned int x = __popc(valuesAndMasks.At(defaultMinutia(), threadIdx.x + blockIdx.y* constsGPU[0].numberCell / 32));
	atomicAdd(&(sum[2 * defaultMinutia() + blockIdx.y]), x);
}

__global__ void createCylinders(CUDAArray<Minutia> minutiae, unsigned int* sum,
	CUDAArray<unsigned int> valuesAndMasks, Cylinder* cylinders)
{
	Cylinder tmpCylinder = Cylinder(valuesAndMasks.AtPtr(blockIdx.x, blockIdx.y* constsGPU[0].numberCell / 32), minutiae.At(0, blockIdx.x).angle, sqrt((float)(sum[blockIdx.x * 2 + blockIdx.y])), constsGPU[0].numberCell / 32);

	cylinders[blockIdx.x * 2 + blockIdx.y] = tmpCylinder;
}


__global__ void createValuesAndMasks(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> valuesAndMasks,
	CUDAArray<float> sumArr)
{
	int lenghtNeighborhood = 0;
	if (defaultX() > 16 || defaultY() > 16 || defaultZ() > 6 || defaultMinutia() > minutiae.Width)
	{
		return;
	}
	if (isValidPoint(&minutiae.At(0, defaultMinutia())))
	{
		char tempValue =
			(stepFunction(sumArr.At(0, defaultMinutia()*constsGPU[0].baseCuboid*constsGPU[0].baseCuboid +
			defaultX()*constsGPU[0].baseCuboid +
			defaultY())));
			tempValue=((tempValue)* ((threadIdx.z + 1) % 2)) + threadIdx.z;
		atomicOr(valuesAndMasks.AtPtr(defaultMinutia(), curIndex()), tempValue << linearizationIndex() % 32);
	}
	else
	{
		atomicOr(valuesAndMasks.AtPtr(defaultMinutia(), curIndex()), 0 << linearizationIndex() % 32);
	}
}

__global__ void getSum( CUDAArray<Minutia> minutiaArr, CUDAArray<float> sumArr)
{
	__shared__ float sumArrShared[100];
	__shared__ float sum[1];
	if ((pointDistance(Point((float)minutiaArr.At(0, threadIdx.x).x, (float)(minutiaArr.At(0, threadIdx.x).y)),
		*getPoint(&minutiaArr.At(0, defaultMinutia()), blockIdx.y, blockIdx.z))) < 3 * constsGPU[0].sigmaLocation &&
		(!equalsMinutae(minutiaArr.AtPtr(0, threadIdx.x), minutiaArr.AtPtr(0, defaultMinutia()))))
	{
		sumArrShared[threadIdx.x] = gaussianLocation(minutiaArr.AtPtr(0, threadIdx.x),
			getPoint(minutiaArr.AtPtr(0, defaultMinutia()), defaultX(), defaultY())) *
			gaussianDirection(minutiaArr.AtPtr(0, defaultMinutia()), minutiaArr.AtPtr(0, threadIdx.x), angleHeight());
		atomicAdd(&sum[0], sumArrShared[threadIdx.x]);

		sumArr.SetAt(0,
			defaultMinutia()*constsGPU[0].baseCuboid*constsGPU[0].baseCuboid +
			blockIdx.y*constsGPU[0].baseCuboid +
			blockIdx.z,sum[0]);
	}
}

void createTemplate(Minutia* minutiae, int lenght, Cylinder** cylinders, int* cylindersLenght)
{
	cudaSetDevice(0);
	Consts *myConst = (Consts*)malloc(sizeof(Consts));
	myConst[0].radius = 70;
	myConst[0].baseCuboid = 16;
	myConst[0].heightCuboid = 6;
	myConst[0].numberCell = myConst[0].baseCuboid *  myConst[0].baseCuboid *  myConst[0].heightCuboid;
	myConst[0].baseCell = (2.0 *  myConst[0].radius) / myConst[0].baseCuboid;
	myConst[0].heightCell = (2 * CUDART_PI) / myConst[0].heightCuboid;
	myConst[0].sigmaLocation = 28.0 / 3;
	myConst[0].sigmaDirection = 2 * CUDART_PI / 9;
	myConst[0].sigmoidParametrPsi = 0.01;
	myConst[0].omega = 50;
	myConst[0].minNumberMinutiae = 2;

	cudaMemcpyToSymbol(constsGPU, myConst, sizeof(Consts));
	cudaCheckError();

	Point* points = (Point*)malloc(lenght * sizeof(Point));
	CUDAArray<Minutia> cudaMinutiae = CUDAArray<Minutia>(minutiae, lenght, 1);
	CUDAArray<Point> cudaPoints = CUDAArray<Point>(points, lenght, 1);
	free(points);
	getPoints << <1, lenght >> >(cudaMinutiae, cudaPoints);
	cudaCheckError();

	int hullLenght = 0;
	Point* hull = (Point*)malloc(lenght*sizeof(Point));
	getConvexHull(cudaPoints.GetData(), lenght, hull, &hullLenght);
	cudaPoints.Dispose();

	Point* extHull = extendHull(hull, hullLenght, myConst[0].omega);
	free(hull);

	int extLenght;
	extLenght = hullLenght * 2;

	cudaMemcpyToSymbol(hullGPU, extHull, sizeof(Point)*extLenght);
	cudaCheckError();

	cudaMemcpyToSymbol(hullLenghtGPU, &extLenght, sizeof(int));
	cudaCheckError();

	bool* isValidMinutiae = (bool*)malloc(lenght*sizeof(bool));
	CUDAArray<bool> cudaIsValidMinutiae = CUDAArray<bool>(isValidMinutiae, lenght, 1);

	getValidMinutiae << <1, lenght >> >(cudaMinutiae, cudaIsValidMinutiae);
	cudaCheckError();

	cudaMinutiae.Dispose();
	cudaIsValidMinutiae.GetData(isValidMinutiae);
	cudaIsValidMinutiae.Dispose();

	int validMinutiaeLenght = 0;
	Minutia* validMinutiae = (Minutia*)malloc(lenght*sizeof(Minutia));
	for (int i = 0; i < lenght; i++)
	{
		if (isValidMinutiae[i])
		{
			validMinutiae[validMinutiaeLenght] = minutiae[i];
			validMinutiaeLenght++;
		}
	}
	free(isValidMinutiae);

	validMinutiae = (Minutia*)realloc(validMinutiae, validMinutiaeLenght*sizeof(Minutia));
	cudaMinutiae = CUDAArray<Minutia>(validMinutiae, validMinutiaeLenght, 1);
	unsigned int* valuesAndMasks = (unsigned int*)malloc(validMinutiaeLenght*sizeof(unsigned int) * 2 * myConst[0].numberCell / 32);
	memset(valuesAndMasks, 0, validMinutiaeLenght*sizeof(unsigned int) * 2 * myConst[0].numberCell / 32);
	CUDAArray<float> cudaSum = CUDAArray<float>(1, validMinutiaeLenght*myConst[0].baseCuboid*myConst[0].baseCuboid);	

	getSum << <dim3(validMinutiaeLenght, myConst[0].baseCuboid, myConst[0].baseCuboid), 100 >> >(cudaMinutiae, cudaSum);
	cudaCheckError();
	CUDAArray <unsigned int> cudaValuesAndMasks = CUDAArray<unsigned int>(valuesAndMasks, 2 * myConst[0].numberCell / 32, validMinutiaeLenght);
	free(valuesAndMasks);

	createValuesAndMasks << < dim3(validMinutiaeLenght, myConst[0].heightCuboid), dim3(myConst[0].baseCuboid, myConst[0].baseCuboid, 2) >> >(cudaMinutiae, cudaValuesAndMasks, cudaSum);
	cudaCheckError();

	cudaSum.Dispose();
	unsigned int* sumArr = (unsigned int*)malloc(2 * validMinutiaeLenght * sizeof(unsigned int));

	memset(sumArr, 0, 2 * validMinutiaeLenght * sizeof(unsigned int));

	unsigned int* cudaSumArr;
	cudaMalloc((void**)&cudaSumArr, 2 * validMinutiaeLenght * sizeof(unsigned int));
	cudaCheckError();
	cudaMemcpy(cudaSumArr, sumArr, 2 * validMinutiaeLenght * sizeof(unsigned int), cudaMemcpyHostToDevice);
	cudaCheckError();

	free(sumArr);
	cudaCheckError();

	createSum << <dim3(validMinutiaeLenght, 2), myConst[0].numberCell / 32 >> >(cudaValuesAndMasks, cudaSumArr);
	cudaCheckError();

	Cylinder* cudaCylinders;
	cudaMalloc((void**)&cudaCylinders, sizeof(Cylinder)*validMinutiaeLenght * 2);
	cudaCheckError();
	createCylinders << <dim3(validMinutiaeLenght, 2), 1 >> >(cudaMinutiae, cudaSumArr, cudaValuesAndMasks, cudaCylinders);
	cudaCheckError();
	
	Cylinder *cylindersTmp = (Cylinder*)malloc(sizeof(Cylinder) * validMinutiaeLenght * 2);
	cudaMemcpy(cylindersTmp, cudaCylinders, sizeof(Cylinder)*validMinutiaeLenght * 2, cudaMemcpyDeviceToHost);
	cudaFree(cudaCylinders);
	for (int i = 0; i < validMinutiaeLenght * 2; i++)
	{
		cylindersTmp[i].deviceToHost();
	}
	cudaValuesAndMasks.Dispose();

	unsigned int* sum = (unsigned int*)malloc(2 * validMinutiaeLenght * sizeof(unsigned int));
	cudaMemcpy(sum, cudaSumArr, 2 * validMinutiaeLenght * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	cudaCheckError();
	Cylinder* validCylinders = (Cylinder*)malloc(2 * validMinutiaeLenght*sizeof(Cylinder));
	unsigned int	maxSum = 0;
	for (int i = 1; i < validMinutiaeLenght * 2; i += 2)
	{
		maxSum = sum[i]> maxSum ? sum[i] : maxSum;
	}
	int validCylindersLenght = 0;
	for (int i = 1; i < validMinutiaeLenght * 2; i += 2)
	{
		if (sum[i] >= 0.75*maxSum)
		{
			validCylinders[validCylindersLenght] = cylindersTmp[i - 1];
			validCylindersLenght++;
			validCylinders[validCylindersLenght] = cylindersTmp[i];
			validCylindersLenght++;
		}
	}
	cudaFree(cudaSumArr);
	validCylinders = (Cylinder*)realloc(validCylinders, validCylindersLenght*sizeof(Cylinder));
	free(cylindersTmp);
	*cylinders = validCylinders;
	*cylindersLenght = validCylindersLenght;
	cudaFree(hullGPU);
	cudaFree(hullLenghtGPU);

}

int main()
{
	int l = 100;
	Minutia* minutiae = (Minutia*)malloc(sizeof(Minutia) * l);
	Minutia tmp;
	for (int i = 0; i < l; i++)
	{
		tmp.x = i + 1;
		tmp.y = (float)(sin((float)(i + 1)));
		tmp.angle = i*0.3;
		minutiae[i] = tmp;
	}
	Cylinder* cylinders = NULL;
	int lenght;
	createTemplate(minutiae, l, &cylinders, &lenght);
	cudaDeviceReset();
	printf("%d", lenght);
	free(minutiae);
}

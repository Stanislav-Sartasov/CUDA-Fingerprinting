#ifndef CUDAFINGERPRINTING_CREATETEMPLATE
#define CUDAFINGERPRINTING_CREATETEMPLATE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "math_constants.h"
#include "VectorHelper.cuh"
#include "CUDAArray.cuh"

struct Consts
{
	char radius;
	char baseCuboid;
	char heightCuboid;
	unsigned int numberCell;
	float baseCell;
	float heightCell;
	float sigmaLocation;
	float sigmaDirection;
	float sigmoidParametrPsi;
	char omega;
	char minNumberMinutiae;
};

struct Minutia
{
	float angle;
	int x;
	int y;
};

struct CylinderMulti
{
public:
	unsigned int values[48];
	float angle;
	float norm;

	CylinderMulti()
	{

	}

	__device__ __host__ CylinderMulti(unsigned int *givenValues, float givenAngle, float givenNorm) :
		angle(givenAngle), norm(givenNorm)
	{
		for (int i = 0; i < 48; i++)
		{
			values[i] = *(givenValues + i*4);
		}
	}
};

__constant__ Consts constsGPU[1];

__device__ float getPointDistance(Point A, Point B);
__device__  Point* getPoint(Minutia *minutiae);
__device__ Minutia** getNeighborhood(CUDAArray<Minutia> *minutiaArr, int *lenghtNeighborhood);
__device__  float angleHeight();
__device__ float gaussian1D(float x);
__device__ float gaussianLocation(Minutia *minutia, Point *point);
__device__ float gaussianDirection(Minutia *middleMinutia, Minutia *minutia, float anglePoint);
__inline__ __device__ bool equalsMinutae(Minutia* firstMinutia, Minutia* secondMinutia);
__device__ bool isValidPoint(Minutia* middleMinutia, Point* hullGPU, int* hullLenghtGPU);
__device__ float sum(Minutia** neighborhood, Minutia* middleMinutia);
__device__ char stepFunction(float value);
void createTemplate(Minutia* minutiae, int lenght, CylinderMulti** cylinders, int* cylindersLenght);
__global__ void createValuesAndMasks(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> valuesAndMasks, Point* hullGPU, int* hullLenghtGPU);
__global__ void getValidMinutiae(CUDAArray<Minutia> minutiae, CUDAArray<bool> isValidMinutiae);
__global__ void getPoints(CUDAArray<Minutia> minutiae, CUDAArray<Point> points);
__global__ void createCylinders(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> sum, CUDAArray<unsigned int> valuesAndMasks, CUDAArray<CylinderMulti> cylinders);
__global__ void createSum(CUDAArray<unsigned int> valuesAndMasks, CUDAArray<unsigned int> sum);

#define defaultX() threadIdx.x+1
#define defaultY() threadIdx.y+1
#define defaultZ() (blockIdx.y+1)
#define defaultMinutia() blockIdx.x

#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess) {\
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(0);\
										}\
}
#define linearizationLenght() constsGPU[0].baseCuboid*constsGPU[0].baseCuboid*constsGPU[0].heightCuboid/32
#define linearizationIndex() (defaultZ()-1)*constsGPU[0].baseCuboid*constsGPU[0].baseCuboid+(defaultY()-1)*constsGPU[0].baseCuboid+defaultX()-1
#define curIndex() linearizationIndex()/32+threadIdx.z*linearizationLenght()
#endif
#ifndef CUDAFINGERPRINTING_CREATETEMPLATE
#define CUDAFINGERPRINTING_CREATETEMPLATE
#include "math_constants.h"
#include "VectorHelper.cuh"

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

struct Minutia
{
	float angle;
	int x;
	int y;
};

__constant__ Consts constsGPU;
Point* hullGPU;

__device__  Point* getPoint(Minutia *minutiae);
__device__ CUDAArray<Minutia*> getNeighborhood(CUDAArray<Minutia> *minutiaArr);
__device__  float angleHeight();
__device__ __host__ float gaussian1D(float x);
__device__ __host__ float gaussianLocation(Minutia *minutia, Point *point);
__device__ float gaussianDirection(Minutia *middleMinutia, Minutia *minutia, float anglePoint);
__inline__ __device__ bool equalsMinutae(Minutia* firstMinutia, Minutia* secondMinutia);
__device__ __host__ bool isValidPoint(Minutia* middleMinutia, Point* hull, int hullLength);
__device__ __host__ float sum(CUDAArray<Minutia*> neighborhood, Minutia* middleMinutia);
__device__ __host__ char stepFunction(float value);
void createTemplate(Minutia* minutiae, int lenght, Cylinder* cylinders, int* cylindersLenght);
__global__ void createValuesAndMasks(CUDAArray<Minutia> minutiae, CUDAArray<unsigned int> values, CUDAArray<unsigned int> masks, Point* hull, int hullLenght);
__global__ void getValidMinutias(CUDAArray<Minutia> minutiae, CUDAArray<bool> isValidMinutiae);
__global__ void getPoints(CUDAArray<Minutia> minutiae, CUDAArray<Point> points, int lenght);

#define defaultX() threadIdx.x+1
#define defaultY() threadIdx.y+1
#define defaultZ() (blockIdx.y+1)*(threadIdx.z+1)
#define defaultMinutia() blockIdx.x

#define linearizationIndex() (defaultZ()-1)*constsGPU.baseCuboid*constsGPU.baseCuboid+(defaultY()-1)*constsGPU.baseCuboid+defaultX()-1
#endif
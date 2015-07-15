#ifndef CUDAFINGERPRINTING_CYLINDERHELPER
#define CUDAFINGERPRINTING_CYLINDERHELPER

#include "CUDAArray.cuh"
#include "stdlib.h"

class Cylinder
{
public:
	unsigned int *values;
	unsigned int valuesCount;
	float angle;
	float norm;
	unsigned int templateIndex;

	Cylinder(unsigned int *givenValues, unsigned int valuesCount, float givenAngle, float givenNorm) :
		angle(givenAngle), norm(givenNorm), values((unsigned int *)malloc(valuesCount * sizeof(unsigned int))) {}
};

class CylinderGPU
{
public:
	CUDAArray<unsigned int> values;
	float angle;
	float norm;
	unsigned int templateIndex;

	CylinderGPU(unsigned int *givenValues, unsigned int valuesCount, float givenAngle, float givenNorm) :
		angle(givenAngle), norm(givenNorm), values(CUDAArray<unsigned int>(givenValues, valuesCount, 1)) {}
};

#endif CUDAFINGERPRINTING_CYLINDERHELPER
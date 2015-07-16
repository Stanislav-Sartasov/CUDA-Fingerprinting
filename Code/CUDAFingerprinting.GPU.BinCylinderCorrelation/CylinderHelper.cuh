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

	Cylinder(unsigned int *givenValues, unsigned int givenValuesCount, float givenAngle, float givenNorm, unsigned int givenTemplateIndex) :
		valuesCount(givenValuesCount), angle(givenAngle), norm(givenNorm), templateIndex(givenTemplateIndex) 
	{
		values = (unsigned int *)malloc(givenValuesCount * sizeof(unsigned int));
		memcpy(values, givenValues, givenValuesCount * sizeof(unsigned int));
	}
};

class CylinderGPU
{
public:
	CUDAArray<unsigned int> *values;
	float angle;
	float norm;
	unsigned int templateIndex;

	CylinderGPU(unsigned int givenValuesCount) 
	{
		CUDAArray<unsigned int> *preValues = new CUDAArray<unsigned int>(givenValuesCount, 1);
		cudaMalloc(&values, sizeof(CUDAArray<unsigned int>));
		cudaMemcpy(values, preValues, sizeof(CUDAArray<unsigned int>), cudaMemcpyHostToDevice);
	}

	CylinderGPU(unsigned int *givenValues, unsigned int givenValuesCount, float givenAngle, float givenNorm, unsigned int givenTemplateIndex) :
		angle(givenAngle), norm(givenNorm), templateIndex(givenTemplateIndex) 
	{
		CUDAArray<unsigned int> *preValues = new CUDAArray<unsigned int>(givenValues, givenValuesCount, 1);
		cudaMalloc(&values, sizeof(CUDAArray<unsigned int>));
		cudaMemcpy(values, preValues, sizeof(CUDAArray<unsigned int>), cudaMemcpyHostToDevice);
	}
};

#endif CUDAFINGERPRINTING_CYLINDERHELPER
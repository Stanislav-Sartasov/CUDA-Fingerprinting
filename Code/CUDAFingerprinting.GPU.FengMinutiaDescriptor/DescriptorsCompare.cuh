#ifndef CUDAFINGERPRINTING_DESCRIPTORSCOMPARE
#define CUDAFINGERPRINTING_DESCRIPTORSCOMPARE

#include "MinutiaStructs.cuh"

__global__ void compareDescriptors(Descriptor* input, Descriptor* current, Descriptor* temp0, Descriptor* temp1,
	float* s, int height, int width, int pitch);

#endif
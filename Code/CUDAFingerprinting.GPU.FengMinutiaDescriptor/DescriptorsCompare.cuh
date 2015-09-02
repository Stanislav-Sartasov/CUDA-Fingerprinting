#ifndef CUDAFINGERPRINTING_DESCRIPTORSCOMPARE
#define CUDAFINGERPRINTING_DESCRIPTORSCOMPARE

#include "MinutiaStructs.cuh"

__global__ void compareDescriptors(Descriptor* input, Descriptor* current, int height, int width, int pitch, float* s,
	int inputNum, int* currentNum);

#endif
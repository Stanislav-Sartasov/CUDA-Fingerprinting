#ifndef CUDAFINGERPRINTING_MINUTIASTRUCTS
#define CUDAFINGERPRINTING_MINUTIASTRUCTS

#include "CUDAArray.cuh"

struct Minutia
{
	float angle;
	int x;
	int y;

	__device__ Minutia(float a, int b, int c) : angle(a), x(b), y(c) {}
};

struct Descriptor
{
	Minutia* minutias;
	Minutia center;

	Descriptor(Minutia* m, Minutia c) : minutias(m), center(c) {}
};

#endif
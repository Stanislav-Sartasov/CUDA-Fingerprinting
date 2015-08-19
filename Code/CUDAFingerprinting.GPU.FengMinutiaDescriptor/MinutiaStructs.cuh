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
	int length;

	Descriptor(Minutia* m, Minutia c, int l) : minutias(m), center(c), length(l) {}
};

#endif
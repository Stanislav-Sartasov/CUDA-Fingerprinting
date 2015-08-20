#ifndef CUDAFINGERPRINTING_MINUTIASTRUCTS
#define CUDAFINGERPRINTING_MINUTIASTRUCTS

#include "CUDAArray.cuh"

struct Minutia
{
	float angle;
	int x;
	int y;
};

struct Descriptor
{
	Minutia* minutias;
	Minutia center;
	int length;
};

#endif
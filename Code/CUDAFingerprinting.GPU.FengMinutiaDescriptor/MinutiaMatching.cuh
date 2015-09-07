#ifndef CUDAFINGERPRINTING_MINUTIAMATCHING
#define CUDAFINGERPRINTING_MINUTIAMATCHING

#include "cuda_runtime.h"
#include "MinutiaHelper.cuh"

__global__ void topElements(float* arrayOfMatrix, int pitch, int inMatrixPitch, float* top, int topSize);

#endif
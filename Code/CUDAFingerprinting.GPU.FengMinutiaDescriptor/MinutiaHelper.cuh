#ifndef CUDAFINGERPRINTING_MINUTIAHELPER
#define CUDAFINGERPRINTING_MINUTIAHELPER

#include "MinutiaStructs.cuh"

__device__ float sqrLength(Minutia m1, Minutia m2);

__global__ void fingerRead(char *dbPath, int dbSize, Minutia **mins, int *minutiaNum);

#endif
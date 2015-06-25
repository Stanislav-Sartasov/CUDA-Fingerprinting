#ifndef CUDAFINGERPRINTING_CONSTSMACROS
#define CUDAFINGERPRINTING_CONSTSMACROS

// macros

#define defaultRow() blockIdx.y*blockDim.y + threadIdx.y

#define defaultColumn() blockIdx.x*blockDim.x + threadIdx.x

#define ceilMod(x, y) (x+y-1)/y

// consts
const int defaultThreadCount = 32;

#endif
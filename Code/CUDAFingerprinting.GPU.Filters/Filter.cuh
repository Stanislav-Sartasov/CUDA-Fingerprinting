#ifndef CUDAFINGEROPRINTING_FILTER
#define CUDAFINGEROPRINTING_FILTER

#include "CUDAArray.cuh"

extern "C"
{
	__declspec(dllexport) CUDAArray<float> MakeGabor16Filters(int angleNum, float* frequencyArr, int frNum);
	__declspec(dllexport) CUDAArray<float> MakeGabor32Filters(int angleNum, float* frequencyArr, int frNum);
	__declspec(dllexport) CUDAArray<float> MakeGaborFilters(int size, int angleNum, float* frequencyArr, int frNum);
}
float* MakeGaussianFilter(int size, float sigma);
#endif
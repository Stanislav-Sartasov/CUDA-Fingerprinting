#ifndef CUDAFINGEROPRINTING_CONVOLUTION
#define CUDAFINGEROPRINTING_CONVOLUTION

#include "CUDAArray.cuh"

void AddArray(CUDAArray<float> source, CUDAArray<float> addition);

void SubtractArray(CUDAArray<float> source, CUDAArray<float> subtract);

void Convolve(CUDAArray<float> target, CUDAArray<float> source, CUDAArray<float> filter);

void ComplexConvolve(CUDAArray<float> targetReal, CUDAArray<float> targetImaginary,
	CUDAArray<float> sourceReal, CUDAArray<float> sourceImaginary,
	CUDAArray<float> filterReal, CUDAArray<float> filterImaginary);

#endif
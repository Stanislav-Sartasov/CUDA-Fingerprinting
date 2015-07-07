#ifndef CUDAFINGEROPRINTING_IMAGEBINARIZATION
#define CUDAFINGEROPRINTING_IMAGEBINARIZATION

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constsmacros.h"
#include "CUDAArray.cuh"

__global__ void ImageBinarization(CUDAArray<int>, int, CUDAArray<int>);

void BinarizateImage(CUDAArray<int>, int, int*);

#endif
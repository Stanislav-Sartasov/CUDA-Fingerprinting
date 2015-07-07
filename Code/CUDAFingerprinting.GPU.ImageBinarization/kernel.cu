
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "Convolution.cuh"
#include "constsmacros.h"
#include "CUDAArray.cuh"
#include "ImageBinarization.cuh"
#include "ImageLoading.cuh"

int main()
{
	int width = 256;
	int height = 364;
	int *image = loadBmp("2_6.bmp", &width, &height);
	// how to test?
	cudaSetDevice(0);
	CUDAArray<int> cudaImg = CUDAArray<int>(image, width, height);
	cudaImg.cudaPtr = image;

	BinarizateImage(cudaImg, 128, image);

	saveBmp("1.bmp", image, width, height);

	cudaFree(cudaImg.cudaPtr); // ?

    return 0;
}


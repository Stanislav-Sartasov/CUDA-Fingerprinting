#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ImageLoading.cuh"
#include "..//CUDAFingerprinting.GPU.OrientationField//OrientationField.cu"

int main()
{
	char filepath[] = "D://Education//CUDA Fingerprinting 2//CUDA-Fingerprinting//Code//1_1.bmp";
	int width, height;
	int* intBmpArray = loadBmp(filepath, &width, &height);
	float* floatBmpArray = (float*)malloc(sizeof(float) * width * height);
	for (int i = 0; i < width * height; i++){
		floatBmpArray[i] = (float)intBmpArray[i];
	}
	// OrientationFieldInBlocks(floatBmpArray, width, height);

	OrientatiobFieldInPixels(floatBmpArray, width, height);

    return 0;
}


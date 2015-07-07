#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "ImageLoading.cuh"
//#include "OrientationField.cu"

int main()
{
	char filepath[] = "D://Education//CUDA Fingerprinting 2//CUDA-Fingerprinting//Code//1_1.bmp";
	int *width, *height;
	int* intBmpArray = loadBmp(filepath, width, height);
	float* floatBmpArray = new float[];
	for (int i = 0; i < sizeof(intBmpArray) / sizeof(intBmpArray[0]); i++){
		floatBmpArray[i] = (float)intBmpArray[i];
	}
	//OrientationField(floatBmpArray, bmpheader->Width, bmpheader->Height);


    return 0;
}


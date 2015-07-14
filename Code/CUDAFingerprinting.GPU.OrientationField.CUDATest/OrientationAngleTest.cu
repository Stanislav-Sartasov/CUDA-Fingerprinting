//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
//#include "ImageLoading.cuh"
//#include "..//CUDAFingerprinting.GPU.OrientationField//OrientationField.cu"
//
//int main()
//{
//	char filepath[] = "C:\\temp\\1.bmp";
//	int width, height;
//	int* intBmpArray = loadBmp(filepath, &width, &height);
//	float* floatBmpArray = (float*)malloc(sizeof(float) * width * height);
//	for (int i = 0; i < width * height; i++){
//		floatBmpArray[i] = (float)intBmpArray[i];
//	}
//	float* orientation;
//	//orientation = OrientationFieldInBlocks(floatBmpArray, width, height);
//
//	orientation = OrientatiobFieldInPixels(floatBmpArray, width, height);
//
//    return 0;
//}
//

//TEST FILE
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
//
#include "Thinning.cuh"
#include "ImageLoading.cuh"
#include "Utils.h"
//

int main()
{
	int width = 0;
	int height = 0;
	int* img = loadBmp(".\\f.bmp", &width, &height);//test file from folder with executable file
	double** skeleton = Thin(intToDoubleArray(img, width, height), width, height);
	double** res = OverlapArrays(skeleton, intToDoubleArray(img, width, height), width, height);
	saveBmp(".\\result.bmp", doubleToIntArray(res, width, height), width, height);
	
	free(img);
	free(skeleton);
	free(res);

	system(".\\result.bmp");

	return 0;
}
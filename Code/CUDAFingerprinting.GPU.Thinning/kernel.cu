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

//#define PROFILERENABLED

#ifdef PROFILERENABLED
#include "cuda_profiler_api.h"
#endif

int main()
{
	cudaSetDevice(0);
	int width = 0;
	int height = 0;
	int* img = loadBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\idealH.bmp", &width, &height);//test file from folder with executable file

#ifdef PROFILERENABLED
	cudaProfilerStart();
#endif
	double** skeleton = Thin(intToDoubleArray(img, width, height), width, height);
#ifdef PROFILERENABLED
	cudaProfilerStop();
#endif
	double** res = OverlapArrays(skeleton, intToDoubleArray(img, width, height), width, height);
	saveBmp("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\resultCUDA.bmp", doubleToIntArray(res, width, height), width, height);

	free(skeleton);
	free(res);
	system("D:\\Ucheba\\Programming\\summerSchool\\Code\\Debug\\resultCUDA.bmp");

	free(img);
	cudaDeviceReset();
	return 0;
}
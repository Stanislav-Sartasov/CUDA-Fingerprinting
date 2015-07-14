#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constsmacros.h"
#include "CUDAArray.cuh"

extern "C"
{
	__declspec(dllexport)  void BinarizeImage(int line, int* src, int* dist, int width, int height);
}
__global__ void ImageBinarization(CUDAArray<int> src, int line, CUDAArray<int> dev_img)
{
	int row = defaultRow();
	int column = defaultColumn();
	dev_img.SetAt(row, column, src.At(row, column) < line ? 0 : 255);
}

void BinarizeImage(int line, int* src, int* dist, int width, int height)
{
	cudaSetDevice(0);
	CUDAArray<int> img = CUDAArray<int>(src, width, height);
	CUDAArray<int> dev_img = CUDAArray<int>(width, height);

	ImageBinarization <<<dim3(ceilMod(img.Width, defaultThreadCount), ceilMod(img.Height, defaultThreadCount)), dim3(defaultThreadCount, defaultThreadCount) >>>(img, line, dev_img);

	dev_img.GetData(dist);

	img.Dispose();
	dev_img.Dispose();
}
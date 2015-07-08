#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constsmacros.h"
#include "CUDAArray.cuh"

extern "C"
{
	__declspec(dllexport)  void BinarizateImage(int line, int* dist, int width, int height);
}
__global__ void ImageBinarization(CUDAArray<int> src, int line, CUDAArray<int> dev_img)
{
	int x = defaultRow();
	int y = defaultColumn();
	dev_img.SetAt(x, y, src.At(x, y) < line ? 0 : 255);
}

void BinarizateImage(int line, int* dist, int width, int height)
{
	cudaSetDevice(0);
	CUDAArray<int> img = CUDAArray<int>(dist, width, height);
	CUDAArray<int> dev_img = CUDAArray<int>(width, height);

	ImageBinarization <<<dim3(ceilMod(img.Width, defaultThreadCount), ceilMod(img.Height, defaultThreadCount)), dim3(defaultThreadCount, defaultThreadCount) >>>(img, line, dev_img);

	dev_img.GetData(dist);

	img.Dispose();
	dev_img.Dispose();
}
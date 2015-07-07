#include "ImageBinarization.cuh"


__global__ void ImageBinarization(CUDAArray<int> src, int line, CUDAArray<int> dev_img)
{
	int x = defaultRow();
	int y = defaultColumn();
	dev_img.SetAt(x, y, src.At(x, y) < line ? 0 : 255);
	/*
	if (src.At(x, y) < line)
	{
	dev_img.SetAt(x, y, 0);
	}
	else
	{
	dev_img.SetAt(x, y, 255);
	}*/
}

 void BinarizateImage(CUDAArray<int> img, int line, int* dist)
{

	CUDAArray<int> dev_img  = CUDAArray<int> (img.Width, img.Height);

	int size = sizeof(int) * img.Height * img.Width;
	cudaMemcpy(dev_img.cudaPtr, dist, size, cudaMemcpyHostToDevice);

	ImageBinarization <<<img.Height , img.Width >>>(dev_img, line, dev_img);
	
	cudaMemcpy(dist, dev_img.cudaPtr, size, cudaMemcpyDeviceToHost);

	cudaFree(dev_img.cudaPtr);
}
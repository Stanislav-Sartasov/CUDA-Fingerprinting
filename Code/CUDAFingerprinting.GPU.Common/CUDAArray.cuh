#ifndef CUDAFINGEROPRINTING_CUDAARRAY
#define CUDAFINGEROPRINTING_CUDAARRAY

#include "cuda_runtime.h"
#include "constsmacros.h"

template<typename T> class CUDAArray
{
private:
	size_t deviceStride;
public:
	T* cudaPtr;
	size_t Height;
	size_t Width;
	size_t Stride;

	CUDAArray();

	CUDAArray(const CUDAArray& arr);

	CUDAArray(T* cpuPtr, int width, int height);

	CUDAArray(int width, int height);

	T* GetData();

	void GetData(T* arr);

	__device__ T At(int row, int column);

	__device__ void SetAt(int row, int column, T value);

	void Dispose();

	~CUDAArray();
};

template class CUDAArray<int>;

template class CUDAArray<float>;

#endif
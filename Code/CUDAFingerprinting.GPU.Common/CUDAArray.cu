#include "CUDAArray.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>


template<typename T> CUDAArray<T>::CUDAArray()
	{
	}

template<typename T> CUDAArray<T>::CUDAArray(const CUDAArray& arr)
	{
		cudaPtr = arr.cudaPtr;
		Height = arr.Height;
		Width = arr.Width;
		Stride = arr.Stride;
		deviceStride = arr.deviceStride;
	}

template<typename T> CUDAArray<T>::CUDAArray(T* cpuPtr, int width, int height)
	{
		Width = width;
		Height = height;
		cudaError_t error = cudaMallocPitch((void**)&cudaPtr, &Stride, Width*sizeof(T), Height);
		error = cudaDeviceSynchronize();
		deviceStride = Stride / sizeof(T);
		error = cudaMemcpy2D(cudaPtr, Stride, cpuPtr, Width*sizeof(T),
			Width*sizeof(T), Height, cudaMemcpyHostToDevice);
		error = cudaDeviceSynchronize();
		error = cudaGetLastError();
	}

template<typename T> CUDAArray<T>::CUDAArray(int width, int height)
	{
		Width = width;
		Height = height;
		cudaError_t error = cudaMallocPitch((void**)&cudaPtr, &Stride, Width*sizeof(T), Height);
		error = cudaDeviceSynchronize();
		deviceStride = Stride / sizeof(T);
	}

template<typename T> T* CUDAArray<T>::GetData()
	{
		T* arr = (T*)malloc(sizeof(T)*Width*Height);
		GetData(arr);
		return arr;
	}

template<typename T> void CUDAArray<T>::GetData(T* arr)
	{
		cudaError_t error = cudaMemcpy2D(arr, Width*sizeof(T), cudaPtr, Stride, Width*sizeof(T), Height, cudaMemcpyDeviceToHost);
		error = cudaDeviceSynchronize();
	}

template<typename T> __device__ T CUDAArray<T>::At(int row, int column)
	{
		return cudaPtr[row*deviceStride + column];
	}

template<typename T> __device__ void CUDAArray<T>::SetAt(int row, int column, T value)
	{
		cudaPtr[row*deviceStride + column] = value;
	}

template<typename T> void CUDAArray<T>::Dispose()
	{
		cudaFree(cudaPtr);
	}

template<typename T> CUDAArray<T>::~CUDAArray()
	{

	}
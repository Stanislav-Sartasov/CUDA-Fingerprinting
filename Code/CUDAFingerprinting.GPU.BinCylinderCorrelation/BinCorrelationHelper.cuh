#ifndef CUDAFINGERPRINTING_BINCORRELATIONHELPER
#define CUDAFINGERPRINTING_BINCORRELATIONHELPER

extern "C"
{
	__declspec(dllexport) float getBinCylinderCorrelation(
		unsigned int cylinderCapacity,
		unsigned int *cudaCylinder1, unsigned int *cudaCylinder2,
		unsigned int *cudaValidities1, unsigned int *cudaValidities2);
}

void printArray1D(unsigned int* arr, unsigned int length);
void printCUDAArray1D(CUDAArray<unsigned int> arr);
void printArray2D(unsigned int* arr, unsigned int width, unsigned int height);
void printCUDAArray2D(CUDAArray<unsigned int> arr);

//__device__ void cudaArrayBitwiseAndDevice(CUDAArray<unsigned int> *fst, CUDAArray<unsigned int> *snd, CUDAArray<unsigned int> *result);
CUDAArray<unsigned int> BitwiseAndArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd);

__device__ void cudaArrayBitwiseXorDevice(CUDAArray<unsigned int> *fst, CUDAArray<unsigned int> *snd, CUDAArray<unsigned int> *result);
CUDAArray<unsigned int> BitwiseXorArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd);

//__device__ void cudaArrayWordNormDevice(CUDAArray<unsigned int> *arr, unsigned int* sum);
unsigned int getOneBitsCount(CUDAArray<unsigned int> arr);
unsigned int getOneBitsCountRaw(unsigned int* arr, unsigned int length);

unsigned int binToInt(char* s);

#endif CUDAFINGERPRINTING_BINCORRELATIONHELPER
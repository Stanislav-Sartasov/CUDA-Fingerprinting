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

__device__ __inline__ void cudaArrayBitwiseAndDevice(CUDAArray<unsigned int> *fst, CUDAArray<unsigned int> *snd, unsigned int *result)
{
	int row = (defaultRow()) % fst->Height;
	int column = (defaultColumn()) % fst->Width;

	unsigned int newValue = fst->At(row, column) & snd->At(row, column);
	result[row * fst->Width + column] = newValue;
}
CUDAArray<unsigned int> BitwiseAndArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd);

__device__ __inline__ void cudaArrayBitwiseXorDevice(CUDAArray<unsigned int> *fst, CUDAArray<unsigned int> *snd, unsigned int *result)
{
	int row = (defaultRow()) % fst->Height;
	int column = (defaultColumn()) % fst->Width;

	unsigned int newValue = fst->At(row, column) ^ snd->At(row, column);
	result[row * fst->Width + column] = newValue;
}
CUDAArray<unsigned int> BitwiseXorArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd);

__device__ __inline__ void cudaArrayWordNormDevice(unsigned int *arr, unsigned int arrHeight, unsigned int arrWidth, unsigned int* sum)
{
	int row = (defaultRow()) % arrHeight;
	int column = (defaultColumn()) % arrWidth;

	unsigned int x = arr[row * arrWidth + column];

	x = __popc(x);

	atomicAdd(sum, x);
}
unsigned int getOneBitsCount(CUDAArray<unsigned int> arr);
unsigned int getOneBitsCountRaw(unsigned int* arr, unsigned int length);

void createCylinderValues(char* src, unsigned int srcLength, unsigned int *res);

#endif CUDAFINGERPRINTING_BINCORRELATIONHELPER
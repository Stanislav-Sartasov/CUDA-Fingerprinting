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
CUDAArray<unsigned int> BitwiseAndArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd);
CUDAArray<unsigned int> BitwiseXorArray(CUDAArray<unsigned int> fst, CUDAArray<unsigned int> snd);
unsigned int getOneBitsCount(CUDAArray<unsigned int> arr);


#endif CUDAFINGERPRINTING_BINCORRELATIONHELPER
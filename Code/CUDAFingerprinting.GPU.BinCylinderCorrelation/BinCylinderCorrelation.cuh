#ifndef CUDAFINGERPRINTING_BINCYLINDERCORRELATION
#define CUDAFINGERPRINTING_BINCYLINDERCORRELATION

extern "C"
{
	__declspec(dllexport) float getBinCylinderCorrelation(
		unsigned int cylinderCapacity,
		unsigned int *cudaCylinder1, unsigned int *cudaCylinder2,
		unsigned int *cudaValidities1, unsigned int *cudaValidities2);
}

float getBinCylinderCorrelation(
	unsigned int cylinderCapacity,
	unsigned int *cudaCylinder1, unsigned int *cudaCylinder2,
	unsigned int *cudaValidities1, unsigned int *cudaValidities2);

#endif CUDAFINGERPRINTING_BINCYLINDERCORRELATION
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "constsmacros.h"
#include "CUDAArray.cuh"
#include <cstring>
#include <time.h>
#include "BinCorrelationHelper.cuh"
#include "BinCylinderCorrelation.cuh"

float getBinCylinderCorrelation(
	unsigned int cylinderCapacity,
	unsigned int *cudaCylinder1, unsigned int *cudaCylinder2,
	unsigned int *cudaValidities1, unsigned int *cudaValidities2)
{
	//printArray1D(cudaValidities1, cylinderCapacity);

	CUDAArray<unsigned int> linearizedCylinder1(cudaCylinder1, cylinderCapacity, 1);
	CUDAArray<unsigned int> linearizedCylinder2(cudaCylinder2, cylinderCapacity, 1);

	CUDAArray<unsigned int> cylinder1Validities(cudaValidities1, cylinderCapacity, 1);
	CUDAArray<unsigned int> cylinder2Validities(cudaValidities2, cylinderCapacity, 1);

	CUDAArray<unsigned int> commonValidities = BitwiseAndArray(cylinder1Validities, cylinder2Validities);
	CUDAArray<unsigned int> c1GivenCommon = BitwiseAndArray(linearizedCylinder1, commonValidities);
	CUDAArray<unsigned int> c2GivenCommon = BitwiseAndArray(linearizedCylinder2, commonValidities);

	//printf("///\n");
	//printCUDAArray1D(cylinder1Validities);
	//printCUDAArray1D(commonValidities);
	//printCUDAArray1D(c1GivenCommon);
	//printCUDAArray1D(c2GivenCommon);
	//printCUDAArray1D(linearizedCylinder2);
	//printf("///\n");

	unsigned int c1GivenCommonOnesCount = getOneBitsCount(c1GivenCommon);
	float c1GivenCommonNorm = sqrt((float)c1GivenCommonOnesCount);
	unsigned int c2GivenCommonOnesCount = getOneBitsCount(c2GivenCommon);
	float c2GivenCommonNorm = sqrt((float)c2GivenCommonOnesCount);

	//printf("%u, %u\n", c1GivenCommonOnesCount, c2GivenCommonOnesCount);
	//printf("%f, %f\n", c1GivenCommonNorm, c2GivenCommonNorm);

	bool matchable = true;

	if (c1GivenCommonNorm + c2GivenCommonNorm == 0) {
		matchable = false;
	}

	float correlation = 0.0f;

	if (matchable) {
		CUDAArray<unsigned int> givenXOR = BitwiseXorArray(c1GivenCommon, c2GivenCommon);
		//printCUDAArray1D(givenXOR);
		unsigned int givenXORBytesCount = getOneBitsCount(givenXOR);
		float givenXORNorm = sqrt((float)givenXORBytesCount);
		correlation = 1 - givenXORNorm / (c1GivenCommonNorm + c2GivenCommonNorm);
	}

	linearizedCylinder1.Dispose();
	linearizedCylinder2.Dispose();
	cylinder1Validities.Dispose();
	cylinder2Validities.Dispose();

	return correlation;
}

unsigned int binToInt(char* s)
{
	return (unsigned int)strtoul(s, NULL, 2);
}

int main()
{
	unsigned int cylinderCapacity = 1;

	unsigned int *cudaCylinder1 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
	unsigned int *cudaCylinder2 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));

	unsigned int *cudaValidities1 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
	unsigned int *cudaValidities2 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));

	// Test 1
	//memset(cudaCylinder1, 255, cylinderCapacity * sizeof(unsigned int));
	//memset(cudaCylinder2, 255, cylinderCapacity * sizeof(unsigned int));
	//memset(cudaValidities1, 255, cylinderCapacity * sizeof(unsigned int));
	//memset(cudaValidities2, 255, cylinderCapacity * sizeof(unsigned int));
	//getBinCylinderCorrelation(cylinderCapacity, cudaCylinder1, cudaCylinder2, cudaValidities1, cudaValidities2);


	// Test 2
	//srand((unsigned int)time(NULL));
	//for (unsigned int i = 0; i < cylinderCapacity; i++) {
	//	cudaCylinder1[i] = rand();
	//	cudaCylinder2[i] = rand();
	//	cudaValidities1[i] = rand();
	//	cudaValidities2[i] = rand();
	//}
	//getBinCylinderCorrelation(cylinderCapacity, cudaCylinder1, cudaCylinder2, cudaValidities1, cudaValidities2);

	// Test 3 (only for cylinderCapacity == 1)

	cudaCylinder1[0] = binToInt("11111111111111111100000000000000");
	cudaValidities1[0] = binToInt("11111111111111111100000000000000");

	cudaCylinder2[0] = binToInt("11010001010100001100000000000000");
	cudaValidities2[0] = binToInt("11011101111100011100000000000000");

	float correlation =
		getBinCylinderCorrelation(cylinderCapacity, cudaCylinder1, cudaCylinder2, cudaValidities1, cudaValidities2);

	printf("Correlation: %f\n", correlation);

	// [end] Test 3

	free(cudaCylinder1);
	free(cudaCylinder2);
	free(cudaValidities1);
	free(cudaValidities2);

	return 0;
}
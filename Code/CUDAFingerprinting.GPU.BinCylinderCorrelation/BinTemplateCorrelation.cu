#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include <stdlib.h>
#include "constsmacros.h"
#include "CUDAArray.cuh"
#include <cstring>
#include <time.h>
#include "CylinderHelper.cuh"
#include <math_constants.h>
#include <math.h>
#include "BinCorrelationHelper.cuh"
#include "BinTemplateCorrelation.cuh"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define MAX_CYLINDERS_PER_TEMPLATE 256
//#define CYLINDER_CELLS_COUNT 255 // Hopefully this define is not necessary

#define QUANTIZED_ANGLES_COUNT 256
#define QUANTIZED_SIMILARITIES_COUNT 64 // Basically buckets count

#define END_OF_LIST -1

#define ANGLE_THRESHOLD 0.52359877f // == PI / 6

#define THREADS_PER_BLOCK_MATRIX_GEN 192
#define THREADS_PER_BLOCK_LSS 64

#define NUM_PAIRS_MIN 11
#define NUM_PAIRS_MAX 13
#define NUM_PAIRS_MU 30
#define NUM_PAIRS_TAU 0.4

__constant__ CUDAArray<CylinderGPU> queryGPU;
__device__ CUDAArray<CylinderGPU> cylindersDbGPU;
__device__ CUDAArray<unsigned int> bucketMatrix;
__constant__ unsigned int queryLength;

__device__ float getAngleDiff(float angle1, float angle2)
{
	float diff = angle1 - angle2;
	return
		diff < -CUDART_PI_F ? diff + 2 * CUDART_PI_F :
		diff >= CUDART_PI_F ? diff - 2 * CUDART_PI_F :
		diff;
}


__global__ void computeLUTSqrt(CUDAArray<float> LUTSqrt)
{
	LUTSqrt.SetAt(threadIdx.x, 1, sqrt((float)threadIdx.x));
}

__global__ void computeLUTNumPairs(CUDAArray<unsigned int> LUTNumPairs)
{
	for (int i = 0; i < LUTNumPairs.Width; i++)
	{
		unsigned int curNumPairs = NUM_PAIRS_MIN + round((NUM_PAIRS_MAX - NUM_PAIRS_MIN)
			/ (1 + expf(-NUM_PAIRS_TAU * MIN(queryLength, i) - NUM_PAIRS_MU)));
		LUTNumPairs.SetAt(i, 1, curNumPairs);
	}
}

__global__ void computeLUTAngles(CUDAArray<int> LUTAngles)
{
	float curAngle = 2 * CUDART_PI_F * threadIdx.x / QUANTIZED_ANGLES_COUNT;
	unsigned int i;
	unsigned int LUTIndex = 0;
	for (i = 0; i < queryGPU.Width; i++)
	{
		if (getAngleDiff(queryGPU.At(i, 1).angle, curAngle) < ANGLE_THRESHOLD
			&& queryGPU.At(i, 1).norm + queryGPU.At(i, 1).norm != 0)
		{
			LUTAngles.SetAt(curAngle, LUTIndex, i);
			LUTIndex++;
		}
	}
	LUTAngles.SetAt(threadIdx.x, LUTIndex, -1);
}

__global__ void computeXorArray(CUDAArray<CylinderGPU> xorArray)
{
	unsigned int row = defaultRow();
	unsigned int column = defaultColumn();

	cudaArrayBitwiseXorDevice(
		queryGPU.At(row, column).values, cylindersDbGPU.At(row, column).values, xorArray.At(row, column).values);
}

__global__ void cumputeLUTPopCountXor(unsigned int *LUTArr, CUDAArray<CylinderGPU> xorArray)
{
	unsigned int row = defaultRow();
	unsigned int column = defaultColumn();
	
	cudaArrayWordNormDevice(xorArray.At(row / 32, column / 32).values, &LUTArr[column * queryLength + row]);
}

__global__ void generateBucketMatrix(CUDAArray<float> LUTSqrt, CUDAArray<int> LUTAngles, CUDAArray<unsigned int> LUTPopCountXor)
{
	unsigned int row = defaultRow();
	unsigned int column = defaultRow();

	CylinderGPU curCylinderDb = cylindersDbGPU.At(row, column);
	float curCylinderDbAngle = cylindersDbGPU.At(row, column).angle;

	unsigned int angleIndex = (unsigned int)(curCylinderDbAngle * QUANTIZED_ANGLES_COUNT / (2 * CUDART_PI_F));
	for (unsigned int i = 0; LUTAngles.At(angleIndex, i) != -1; i++)
	{
		unsigned int curQueryIndex = LUTAngles.At(angleIndex, i);
		CylinderGPU curCylinderQuery = queryGPU.At(curQueryIndex, 1);

		unsigned int bucketIndex = (unsigned int)floor(LUTSqrt.At(LUTPopCountXor.At(curQueryIndex, 1)
			/ (curCylinderDb.norm + curCylinderQuery.norm) * QUANTIZED_SIMILARITIES_COUNT, 1));

		bucketMatrix.SetAt(curCylinderDb.templateIndex, bucketIndex, bucketMatrix.At(curCylinderDb.templateIndex, bucketIndex) + 1);
	}
}

__global__ void computeLSS(
	CUDAArray<unsigned int> LUTTemplateDbLengths, CUDAArray<unsigned int>LUTNumPairs, CUDAArray<float> similarityRates)
{
	// Dynamic allocation will probably be better
	__shared__ unsigned int bucketSubmatrixPerBlock[QUANTIZED_SIMILARITIES_COUNT][THREADS_PER_BLOCK_LSS];

	// Index of the first DB template of the current block
	unsigned int firstBlockTemplateIndex = THREADS_PER_BLOCK_LSS * blockIdx.x;

	// Copy appropriate part of bucketMatrix to bucketSubmatrixPerBlock
	for (int j = 0; j < THREADS_PER_BLOCK_LSS; j++)
	{
		bucketSubmatrixPerBlock[j][threadIdx.x] = bucketMatrix.At(firstBlockTemplateIndex + j, threadIdx.x);
	}

	__syncthreads();

	unsigned int curTemplateIndex = firstBlockTemplateIndex + threadIdx.x; // == defaultColumn() (hopefully)

	unsigned int numPairs = LUTNumPairs.At(MIN(LUTTemplateDbLengths.At(curTemplateIndex, 1), queryLength), 1);
	int sum = 0, t = numPairs, i = 0;
	while (i < QUANTIZED_SIMILARITIES_COUNT && t > 0)
	{
		unsigned int curBucketValue = bucketSubmatrixPerBlock[threadIdx.x][i];
		sum += MIN(curBucketValue, t);
		t -= MIN(curBucketValue, t);
		i++;
	}
	sum += t * QUANTIZED_SIMILARITIES_COUNT;

	similarityRates.SetAt(curTemplateIndex, 1, 1 - sum / (numPairs * QUANTIZED_SIMILARITIES_COUNT));
}

void convertToCylindersGPU(Cylinder *cylinders, unsigned int cylindersCount, CUDAArray<CylinderGPU> *cylindersGPU)
{
	CylinderGPU *cylindersGPUarr = (CylinderGPU *)malloc(cylindersCount * sizeof(CylinderGPU));
	for (unsigned int i = 0; i < cylindersCount; i++)
	{
		Cylinder *curCylinder = &(cylinders[i]);
		cylindersGPUarr[i] = CylinderGPU(
			curCylinder->values, curCylinder->valuesCount, curCylinder->angle, curCylinder->norm);		
	}

	*cylindersGPU = CUDAArray<CylinderGPU>(cylindersGPUarr, cylindersCount, 1);
}

float * getBinTemplateSimilarity(
	Cylinder *query, unsigned int queryLength,
	Cylinder *cylindersDb, unsigned int cylinderDbCount,
	unsigned int *templateDbLengths, unsigned int templateDbCount)
{
	CUDAArray<CylinderGPU> preQueryGPU = CUDAArray<CylinderGPU>(queryLength, 1);
	convertToCylindersGPU(query, queryLength, &preQueryGPU);
	cudaMemcpy(&queryGPU, &preQueryGPU, sizeof(CUDAArray<CylinderGPU>), cudaMemcpyHostToDevice);

	CUDAArray<CylinderGPU> preCylindersDbGPU = CUDAArray<CylinderGPU>(cylinderDbCount, 1);
	convertToCylindersGPU(cylindersDb, cylinderDbCount, &preCylindersDbGPU);
	cudaMemcpy(&cylindersDbGPU, &preCylindersDbGPU, sizeof(CUDAArray<CylinderGPU>), cudaMemcpyHostToDevice);

	CUDAArray<unsigned int> LUTTemplateDbLengths(templateDbLengths, templateDbCount, 1);

	// It's supposed to work only when all the cylinders have the same length, index = 0 WLOG
	unsigned int cylinderCellsCount = cylindersDb[0].valuesCount;

	// 0 through cylinderCellsCount (population count values)
	CUDAArray<float> LUTSqrt(cylinderCellsCount + 1, 1);
	computeLUTSqrt << <1, cylinderCellsCount + 1 >> >(LUTSqrt);

	CUDAArray<unsigned int> LUTNumPairs(cylinderCellsCount, 1);
	computeLUTNumPairs << <1, cylinderCellsCount >> >(LUTNumPairs);

	CUDAArray<int> LUTAngles(QUANTIZED_ANGLES_COUNT, queryLength + 1);
	computeLUTAngles << <1, QUANTIZED_ANGLES_COUNT >> >(LUTAngles);

	unsigned int *d_LUTArr;
	cudaMalloc((void **)&d_LUTArr, cylinderDbCount * queryLength * sizeof(unsigned int));
	
	CUDAArray<CylinderGPU> xorArray(queryLength, cylinderDbCount);
	cumputeLUTPopCountXor << <cylinderDbCount, queryLength >> >(d_LUTArr, xorArray);

	unsigned int *h_LUTArr = new unsigned int[cylinderDbCount * queryLength];	
	cudaMemcpy(h_LUTArr, d_LUTArr, cylinderDbCount * queryLength * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	CUDAArray<unsigned int> LUTPopCountXor(h_LUTArr, queryLength, cylinderDbCount);

	CUDAArray<unsigned int> preBucketMatrix = CUDAArray<unsigned int>(QUANTIZED_SIMILARITIES_COUNT, templateDbCount);
	cudaMemcpy(&bucketMatrix, &preBucketMatrix, sizeof(CUDAArray<unsigned int>), cudaMemcpyHostToDevice);
	generateBucketMatrix << <ceilMod(cylinderDbCount, THREADS_PER_BLOCK_MATRIX_GEN), THREADS_PER_BLOCK_MATRIX_GEN >> >(
		LUTSqrt, LUTAngles, LUTPopCountXor);

	CUDAArray<float> similaritiesVector(templateDbCount, 1);
	computeLSS << <ceilMod(templateDbCount, THREADS_PER_BLOCK_LSS), THREADS_PER_BLOCK_LSS >> >
		(LUTTemplateDbLengths, LUTNumPairs, similaritiesVector);

	LUTSqrt.Dispose();
	LUTAngles.Dispose();
	LUTNumPairs.Dispose();
	bucketMatrix.Dispose();

	float* result = similaritiesVector.GetData();

	similaritiesVector.Dispose();

	return result;
}
//
//unsigned int binToInt(char* s)
//{
//	return (unsigned int)strtoul(s, NULL, 2);
//}
//
//int main()
//{
//	unsigned int cylinderCapacity = 1;
//
//	unsigned int *cudaCylinder1 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//	unsigned int *cudaCylinder2 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//
//	unsigned int *cudaValidities1 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//	unsigned int *cudaValidities2 = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//
//	// Test 1
//	//memset(cudaCylinder1, 255, cylinderCapacity * sizeof(unsigned int));
//	//memset(cudaCylinder2, 255, cylinderCapacity * sizeof(unsigned int));
//	//memset(cudaValidities1, 255, cylinderCapacity * sizeof(unsigned int));
//	//memset(cudaValidities2, 255, cylinderCapacity * sizeof(unsigned int));
//	//getBinCylinderCorrelation(cylinderCapacity, cudaCylinder1, cudaCylinder2, cudaValidities1, cudaValidities2);
//
//
//	// Test 2
//	//srand((unsigned int)time(NULL));
//	//for (unsigned int i = 0; i < cylinderCapacity; i++) {
//	//	cudaCylinder1[i] = rand();
//	//	cudaCylinder2[i] = rand();
//	//	cudaValidities1[i] = rand();
//	//	cudaValidities2[i] = rand();
//	//}
//	//getBinCylinderCorrelation(cylinderCapacity, cudaCylinder1, cudaCylinder2, cudaValidities1, cudaValidities2);
//
//	// Test 3 (only for cylinderCapacity == 1)
//
//	cudaCylinder1[0] = binToInt("11111111111111111100000000000000");
//	cudaValidities1[0] = binToInt("11111111111111111100000000000000");
//
//	cudaCylinder2[0] = binToInt("11010001010100001100000000000000");
//	cudaValidities2[0] = binToInt("11011101111100011100000000000000");
//
//	float correlation =
//		getBinCylinderCorrelation(cylinderCapacity, cudaCylinder1, cudaCylinder2, cudaValidities1, cudaValidities2);
//
//	printf("Correlation: %f\n", correlation);
//
//	// [end] Test 3
//
//	free(cudaCylinder1);
//	free(cudaCylinder2);
//	free(cudaValidities1);
//	free(cudaValidities2);
//
//	return 0;
//}
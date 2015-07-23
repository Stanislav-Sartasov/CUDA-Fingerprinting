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
#include "BinTemplateCorrelation.cuh"

#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError(); \
	if (e != cudaSuccess) {\
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(0);\
	}\
}

// Getting element of CUDAArray of unsigned integers without .At() (used with bucketMatrix & new xorArray (hopefully))
#define UINT_ARRAY_AT(arr, row, column) (arr.cudaPtr[(row) * (arr.Stride / sizeof(unsigned int)) + column])

// Same with CUDAArray of cylinders (used with xorArray)
#define CYLINDER_ARRAY_AT(arr, row, column) (arr.cudaPtr[(row) * (arr.Stride / sizeof(CUDAArray<unsigned int>)) + column])

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define END_OF_LIST -1

__constant__ CUDAArray<CylinderGPU> queryGPU;
__device__ CUDAArray<CylinderGPU> cylinderDbGPU;
__device__ CUDAArray<unsigned int> bucketMatrix;
__constant__ unsigned int queryLengthGlobal;

CUDAArray<float> LUTSqrt;
CUDAArray<unsigned int> LUTNumPairs;
CUDAArray<unsigned int> LUTTemplateDbLengths;

CUDAArray<float> similaritiesVector;

__constant__ unsigned int xorArrayCellsCount;

size_t freeMemory, totalMemory;

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
	LUTSqrt.SetAt(0, threadIdx.x, sqrt((float)threadIdx.x));
}

__global__ void computeLUTNumPairs(CUDAArray<unsigned int> LUTNumPairs)
{
	int threadIndex = defaultColumn();
	unsigned int curNumPairs = NUM_PAIRS_MIN + round((NUM_PAIRS_MAX - NUM_PAIRS_MIN)
		/ (1 + expf(-1 * NUM_PAIRS_TAU * ((int)MIN(queryLengthGlobal, threadIndex) - NUM_PAIRS_MU)))); // Amazingly important cast
	LUTNumPairs.SetAt(0, threadIndex, curNumPairs);
}

__global__ void computeLUTAngles(CUDAArray<int> LUTAngles)
{
	float curAngle = 2 * CUDART_PI_F * threadIdx.x / QUANTIZED_ANGLES_COUNT;
	unsigned int i;
	unsigned int LUTIndex = 0;
	for (i = 0; i < queryGPU.Width; i++)
	{
		float diff = getAngleDiff(queryGPU.At(0, i).angle, curAngle);
		if (diff < ANGLE_THRESHOLD)
		{
			LUTAngles.SetAt(threadIdx.x, LUTIndex, i);
			LUTIndex++;
		}
	}
	LUTAngles.SetAt(threadIdx.x, LUTIndex, END_OF_LIST);
}


__global__ void generateBucketMatrix(CUDAArray<float> LUTSqrt, CUDAArray<int> LUTAngles)
{
	unsigned int curCylinderIndex = defaultColumn();

	if (cylinderDbGPU.Width > curCylinderIndex)
	{
		CylinderGPU curCylinderDb = cylinderDbGPU.At(0, curCylinderIndex);
		float curCylinderDbAngle = curCylinderDb.angle;

		unsigned int angleIndex = (unsigned int)(curCylinderDbAngle * QUANTIZED_ANGLES_COUNT / (2 * CUDART_PI_F));
		for (unsigned int i = 0; LUTAngles.At(angleIndex, i) != -1; i++)
		{
			unsigned int curQueryIndex = LUTAngles.At(angleIndex, i);
			CylinderGPU curCylinderQuery = queryGPU.At(0, curQueryIndex);

			unsigned int xor[CYLINDER_CELLS_COUNT];
			unsigned int* dbPtr = curCylinderDb.values->cudaPtr;
			unsigned int* queryPtr = curCylinderQuery.values->cudaPtr;
			unsigned int j;
			for (j = 0; j < CYLINDER_CELLS_COUNT; j++) {
				xor[j] = dbPtr[j] ^ queryPtr[j];
			}
			int sum = 0;
			for (j = 0; j < CYLINDER_CELLS_COUNT; j++) {
				sum += __popc(xor[j]);
			}

			//unsigned int lutPopCountXor = LUTPopCountXor.At(curCylinderIndex, curQueryIndex);
			//float lutSqrt = LUTSqrt.At(0, lutPopCountXor);
			float lutSqrt = LUTSqrt.At(0, sum);
			float x = lutSqrt / (curCylinderDb.norm + curCylinderQuery.norm) * QUANTIZED_SIMILARITIES_COUNT; // local similarity inverse (without 1 - ...)
			unsigned int bucketIndex = (unsigned int)floor(x);

			if (bucketIndex == QUANTIZED_SIMILARITIES_COUNT)
			{
				bucketIndex--;
			}

			atomicAdd(&UINT_ARRAY_AT(bucketMatrix, curCylinderDb.templateIndex, bucketIndex), 1);
		}
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
	int submatrixHeight =
		QUANTIZED_SIMILARITIES_COUNT % THREADS_PER_BLOCK_LSS == 0 ? THREADS_PER_BLOCK_LSS :
		QUANTIZED_SIMILARITIES_COUNT % THREADS_PER_BLOCK_LSS;
	for (int j = 0; j < submatrixHeight; j++)
	{
		bucketSubmatrixPerBlock[j][threadIdx.x] = bucketMatrix.At(firstBlockTemplateIndex + j, threadIdx.x);
	}

	__syncthreads();

	unsigned int threadIndex = defaultColumn();
	if (bucketMatrix.Height > threadIndex)
	{
		unsigned int numPairs = LUTNumPairs.At(0, MIN(LUTTemplateDbLengths.At(0, threadIndex), queryLengthGlobal));
		int sum = 0, t = numPairs, i = 0;
		while (i < QUANTIZED_SIMILARITIES_COUNT && t > 0)
		{
			unsigned int curBucketValue = bucketSubmatrixPerBlock[threadIdx.x][i];
			sum += MIN(curBucketValue, t) * i;
			t -= MIN(curBucketValue, t);
			i++;
		}
		sum += t * QUANTIZED_SIMILARITIES_COUNT;

		similarityRates.SetAt(0, threadIndex, 1.0 - (float)sum / (numPairs * QUANTIZED_SIMILARITIES_COUNT));
	}
}

void convertToCylindersGPU(Cylinder *cylinders, unsigned int cylindersCount, CUDAArray<CylinderGPU> *cylindersGPU)
{
	CylinderGPU *cylindersGPUarr = (CylinderGPU *)malloc(cylindersCount * sizeof(CylinderGPU));
	for (unsigned int i = 0; i < cylindersCount; i++)
	{
		Cylinder *curCylinder = &(cylinders[i]);
		cylindersGPUarr[i] = CylinderGPU(
			curCylinder->values, curCylinder->valuesCount, curCylinder->angle, curCylinder->norm, curCylinder->templateIndex);
	}

	*cylindersGPU = CUDAArray<CylinderGPU>(cylindersGPUarr, cylindersCount, 1);
}

void printAngles(int* arr, int width, int height)
{
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			printf("%d ", arr[i * width + j]);
		}
		printf("\n");
	}
}

void printCUDAAngles(CUDAArray<int> arr)
{
	printf("Print CUDAArray 2D\n");
	printAngles(arr.GetData(), arr.Width, arr.Height);
	printf("[end] Print CUDAArray 2D\n");
}

void initMCC(
	Cylinder *cylinderDb, unsigned int cylinderDbCount,
	unsigned int *templateDbLengths, unsigned int templateDbCount)
{
	//printf("Started getting similarities\n");
	cudaSetDevice(0);

	//printf("Started copying DB\n");
	CUDAArray<CylinderGPU> preCylinderDbGPU = CUDAArray<CylinderGPU>(cylinderDbCount, 1);
	convertToCylindersGPU(cylinderDb, cylinderDbCount, &preCylinderDbGPU);
	cudaCheckError();
	cudaMemcpyToSymbol(cylinderDbGPU, &preCylinderDbGPU, sizeof(CUDAArray<CylinderGPU>));
	cudaCheckError();
	cudaDeviceSynchronize();
	//printf("Ended copying DB\n");

	printf("Started computing LUTs\n");

	LUTTemplateDbLengths = CUDAArray<unsigned int>(templateDbLengths, templateDbCount, 1);

	// It's supposed to work only when all the cylinders have the same length, index = 0 WLOG
	unsigned int cylinderCellsCount = cylinderDb[0].valuesCount;

	// 0 through cylinderCellsCount (population count values)
	LUTSqrt = CUDAArray<float>(cylinderCellsCount * sizeof(unsigned int)* 8 + 1, 1);
	computeLUTSqrt << <1, cylinderCellsCount * sizeof(unsigned int)* 8 + 1 >> >(LUTSqrt);

	LUTNumPairs = CUDAArray<unsigned int>(MAX_CYLINDERS_PER_TEMPLATE, 1);
	computeLUTNumPairs << <1, MAX_CYLINDERS_PER_TEMPLATE >> >(LUTNumPairs);

	similaritiesVector = CUDAArray<float>(templateDbCount, 1);
}

float * processMCC(
	Cylinder *query, unsigned int queryLength,
	unsigned int cylinderDbCount, unsigned int templateDbCount)
{
	clock_t startMethod = clock();
	printf("Start method\n");

	cudaMemcpyToSymbol(queryLengthGlobal, &queryLength, sizeof(unsigned int));
	cudaCheckError();

	CUDAArray<CylinderGPU> preQueryGPU = CUDAArray<CylinderGPU>(queryLength, 1);
	convertToCylindersGPU(query, queryLength, &preQueryGPU);
	cudaMemcpyToSymbol(queryGPU, &preQueryGPU, sizeof(CUDAArray<CylinderGPU>));
	cudaCheckError();
	
	int *preLUTAngles = (int *)malloc((queryLength + 1) * QUANTIZED_ANGLES_COUNT * sizeof(int));
	memset(preLUTAngles, 0, (queryLength + 1) * QUANTIZED_ANGLES_COUNT * sizeof(int));
	CUDAArray<int> LUTAngles = CUDAArray<int>(preLUTAngles, queryLength + 1, QUANTIZED_ANGLES_COUNT);
	computeLUTAngles << <1, QUANTIZED_ANGLES_COUNT >> >(LUTAngles);

	// Hopefully this works as well as the previous version with zeroMatrix (it seems like it doesn't :( )
	//CUDAArray<unsigned int> preBucketMatrix = CUDAArray<unsigned int>(QUANTIZED_SIMILARITIES_COUNT, templateDbCount);
	//cudaMemset2D(preBucketMatrix.cudaPtr, preBucketMatrix.Stride, 0, QUANTIZED_SIMILARITIES_COUNT, templateDbCount);
	//cudaMemcpyToSymbol(bucketMatrix, &preBucketMatrix, sizeof(CUDAArray<unsigned int>));
	//cudaCheckError();

	unsigned int* zeroMatrix = (unsigned int *)malloc(QUANTIZED_SIMILARITIES_COUNT * templateDbCount * sizeof(unsigned int));
	memset(zeroMatrix, 0, QUANTIZED_SIMILARITIES_COUNT * templateDbCount * sizeof(unsigned int));
	CUDAArray<unsigned int> preBucketMatrix = CUDAArray<unsigned int>(zeroMatrix, QUANTIZED_SIMILARITIES_COUNT, templateDbCount);
	free(zeroMatrix);
	cudaMemcpyToSymbol(bucketMatrix, &preBucketMatrix, sizeof(CUDAArray<unsigned int>));
	cudaCheckError();

	generateBucketMatrix << <ceilMod(cylinderDbCount, THREADS_PER_BLOCK_MATRIX_GEN), THREADS_PER_BLOCK_MATRIX_GEN >> >(
		LUTSqrt, LUTAngles);
	
	computeLSS << <ceilMod(templateDbCount, THREADS_PER_BLOCK_LSS), THREADS_PER_BLOCK_LSS >> >
		(LUTTemplateDbLengths, LUTNumPairs, similaritiesVector);

	cudaDeviceSynchronize();
	clock_t end = clock();

	printf("Method time: %ld\n", end - startMethod);

	float *result = similaritiesVector.GetData();

	return result;
}

//int main()
//{
//	unsigned int cylinderCapacity = 1;
//
//	unsigned int *cylinder0Values = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//	unsigned int *cylinder1Values = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//	unsigned int *cylinder2Values = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
//
//	createCylinderValues("00000000000000000000000000000000", 32, cylinder0Values);
//	createCylinderValues("11111111111111111100000000000000", 32, cylinder1Values);
//	createCylinderValues("11010001010100001100000000000000", 32, cylinder2Values);
//
//	Cylinder cylinder0 =
//		Cylinder(cylinder0Values, cylinderCapacity, CUDART_PI_F / 6, sqrt((float)getOneBitsCountRaw(cylinder0Values, cylinderCapacity)), 0);
//	Cylinder cylinder1_0 =
//		Cylinder(cylinder1Values, cylinderCapacity, CUDART_PI_F / 4, sqrt((float)getOneBitsCountRaw(cylinder1Values, cylinderCapacity)), 0);
//	Cylinder cylinder1_1 =
//		Cylinder(cylinder1Values, cylinderCapacity, CUDART_PI_F / 4, sqrt((float)getOneBitsCountRaw(cylinder1Values, cylinderCapacity)), 1);
//	Cylinder cylinder2_1 =
//		Cylinder(cylinder2Values, cylinderCapacity, CUDART_PI_F / 3, sqrt((float)getOneBitsCountRaw(cylinder2Values, cylinderCapacity)), 1);
//	Cylinder cylinder2_2 =
//		Cylinder(cylinder2Values, cylinderCapacity, CUDART_PI_F / 3, sqrt((float)getOneBitsCountRaw(cylinder2Values, cylinderCapacity)), 2);
//
//	Cylinder db[] = { cylinder1_0, cylinder1_1, cylinder2_1, cylinder2_2, cylinder2_2, cylinder2_2, cylinder2_2 };
//	Cylinder query[] = { cylinder2_2 }; // Template index hopefully doesn't matter here
//	unsigned int dbTemplateLengths[] = { 1, 2, 4 };
//	unsigned int dbTemplateCount = sizeof(dbTemplateLengths) / sizeof(unsigned int);
//
//	float *similarities = getBinTemplateSimilarities(
//		query, sizeof(query) / sizeof(Cylinder),
//		db, sizeof(db) / sizeof(Cylinder),
//		dbTemplateLengths, dbTemplateCount);
//
//	printf("Similarities:\n");
//	for (unsigned int i = 0; i < dbTemplateCount; i++)
//	{
//		printf("%f%s", similarities[i], (i == dbTemplateCount - 1 ? "" : ", "));
//	}
//	printf("\n");
//
//	// [end] Test 3
//
//	free(cylinder0Values);
//	free(cylinder1Values);
//	free(cylinder2Values);
//
//	return 0;
//}
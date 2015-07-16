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

#define cudaCheckError() { \
cudaError_t e = cudaGetLastError();   \
if (e != cudaSuccess) {  \
printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
exit(0);\
	}\
}

#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define MAX_CYLINDERS_PER_TEMPLATE 256
#define CYLINDER_CELLS_COUNT 1 // Hopefully this define is not necessary

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
__constant__ unsigned int queryLengthGlobal;

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
			/ (1 + expf(-NUM_PAIRS_TAU * MIN(queryLengthGlobal, i) - NUM_PAIRS_MU)));
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
		if (getAngleDiff(queryGPU.At(i, 0).angle, curAngle) < ANGLE_THRESHOLD)
		{
			LUTAngles.SetAt(curAngle, LUTIndex, i);
			int num = LUTAngles.At(curAngle, LUTIndex);
			LUTIndex++;
		}
	}
	LUTAngles.SetAt(threadIdx.x, LUTIndex, -1);
}

__global__ void computeXorArray(CUDAArray<CylinderGPU> xorArray)
{
	unsigned int threadIndex = defaultColumn();

	if (xorArray.Width * xorArray.Height * CYLINDER_CELLS_COUNT > threadIndex) {
		unsigned int xorArrayRow = threadIndex / CYLINDER_CELLS_COUNT;
		unsigned int xorArrayColumn = threadIndex % CYLINDER_CELLS_COUNT;
		cudaArrayBitwiseXorDevice(
			cylindersDbGPU.At(xorArrayRow, 0).values, queryGPU.At(xorArrayColumn, 0).values,
			xorArray.At(xorArrayRow, xorArrayColumn).values);
	}
}

__global__ void cumputeLUTPopCountXor(unsigned int *LUTArr, CUDAArray<CylinderGPU> xorArray)
{
	unsigned int threadIndex = defaultColumn();

	if (xorArray.Width * xorArray.Height * CYLINDER_CELLS_COUNT > threadIndex) {
		unsigned int xorArrayRow = threadIndex / CYLINDER_CELLS_COUNT;
		unsigned int xorArrayColumn = threadIndex % CYLINDER_CELLS_COUNT;
		cudaArrayWordNormDevice(
			xorArray.At(xorArrayRow, xorArrayColumn).values, 
			&LUTArr[xorArrayColumn * queryLengthGlobal + xorArrayRow]);
	}
}

__global__ void generateBucketMatrix(CUDAArray<float> LUTSqrt, CUDAArray<int> LUTAngles, CUDAArray<unsigned int> LUTPopCountXor)
{
	unsigned int row = defaultRow();
	unsigned int column = defaultRow();

	CylinderGPU curCylinderDb = cylindersDbGPU.At(row, column);
	float curCylinderDbAngle = curCylinderDb.angle;

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

	unsigned int numPairs = LUTNumPairs.At(MIN(LUTTemplateDbLengths.At(curTemplateIndex, 1), queryLengthGlobal), 1);
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
			curCylinder->values, curCylinder->valuesCount, curCylinder->angle, curCylinder->norm, curCylinder->templateIndex);		
	}

	*cylindersGPU = CUDAArray<CylinderGPU>(cylindersGPUarr, cylindersCount, 1);
}

__global__ void checkGPUVars(CylinderGPU *cylinder, unsigned int* result)
{

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


float * getBinTemplateSimilarities(
	Cylinder *query, unsigned int queryLength,
	Cylinder *cylindersDb, unsigned int cylinderDbCount,
	unsigned int *templateDbLengths, unsigned int templateDbCount)
{
	cudaSetDevice(0);
	CUDAArray<CylinderGPU> preQueryGPU = CUDAArray<CylinderGPU>(queryLength, 1);
	convertToCylindersGPU(query, queryLength, &preQueryGPU);

	// Why not the address of queryGPU? o_o yet it seems to work
	cudaError_t error = cudaMemcpyToSymbol(queryGPU, &preQueryGPU, sizeof(CUDAArray<CylinderGPU>));
	cudaCheckError();
	//CylinderGPU *cylinder;
	//cudaMalloc(&cylinder, sizeof(Cylinder));
	//unsigned int *result;
	//unsigned int *h_result = (unsigned int *)malloc(sizeof(unsigned int));
	//cudaMalloc(&result, sizeof(unsigned int));
	//checkGPUVars << <1, 1 >> >(cylinder, result);
	//cudaMemcpy(h_result, result, sizeof(unsigned int), cudaMemcpyDeviceToHost);

	CUDAArray<CylinderGPU> preCylindersDbGPU = CUDAArray<CylinderGPU>(cylinderDbCount, 1);
	convertToCylindersGPU(cylindersDb, cylinderDbCount, &preCylindersDbGPU);
	cudaError_t error1 = cudaMemcpyToSymbol(cylindersDbGPU, &preCylindersDbGPU, sizeof(CUDAArray<CylinderGPU>));
	cudaCheckError();

	CUDAArray<unsigned int> LUTTemplateDbLengths(templateDbLengths, templateDbCount, 1);

	// It's supposed to work only when all the cylinders have the same length, index = 0 WLOG
	unsigned int cylinderCellsCount = cylindersDb[0].valuesCount;

	// 0 through cylinderCellsCount (population count values)
	CUDAArray<float> LUTSqrt(cylinderCellsCount + 1, 1);
	computeLUTSqrt << <1, cylinderCellsCount + 1 >> >(LUTSqrt);

	CUDAArray<unsigned int> LUTNumPairs(cylinderCellsCount, 1);
	computeLUTNumPairs << <1, cylinderCellsCount >> >(LUTNumPairs);

	int *preLUTAngles = (int *)malloc((queryLength + 1) * QUANTIZED_ANGLES_COUNT * sizeof(int));
	//memset(&preLUTAngles, 0, (queryLength + 1) * QUANTIZED_ANGLES_COUNT * sizeof(int)); // Why does it overwrite everything around?
	for (unsigned int i = 0; i < (queryLength + 1) * QUANTIZED_ANGLES_COUNT; i++) {
		preLUTAngles[i] = 0;
	}
	CUDAArray<int> LUTAngles(preLUTAngles, queryLength + 1, QUANTIZED_ANGLES_COUNT);
	//printCUDAAngles(LUTAngles);
	computeLUTAngles << <1, QUANTIZED_ANGLES_COUNT >> >(LUTAngles);

	unsigned int *d_LUTArr;
	cudaMalloc((void **)&d_LUTArr, cylinderDbCount * queryLength * sizeof(unsigned int));
	CylinderGPU *xorArrayCylindersGPU = (CylinderGPU *)malloc(cylinderDbCount * queryLength);
	for (unsigned int i = 0; i < cylinderDbCount * queryLength; i++) {
		xorArrayCylindersGPU[i] = CylinderGPU(cylinderCellsCount);
	}
	CUDAArray<CylinderGPU> xorArray(xorArrayCylindersGPU, queryLength, cylinderDbCount);
	cudaCheckError();
	cudaDeviceSynchronize();
	printCUDAAngles(LUTAngles);
	computeXorArray << <ceilMod(cylinderDbCount * queryLength * cylinderCellsCount, defaultThreadCount), defaultThreadCount >> >(xorArray);
	cudaDeviceSynchronize();
	cudaCheckError();
	printf("Got Xor\n");

	cumputeLUTPopCountXor << <cylinderDbCount * cylinderCellsCount, queryLength >> >(d_LUTArr, xorArray);

	unsigned int *h_LUTArr = new unsigned int[cylinderDbCount * queryLength];	
	cudaMemcpy(h_LUTArr, d_LUTArr, cylinderDbCount * queryLength * sizeof(unsigned int), cudaMemcpyDeviceToHost);
	CUDAArray<unsigned int> LUTPopCountXor(h_LUTArr, queryLength, cylinderDbCount);
	
	unsigned int *zeroMatrix = (unsigned int *)malloc(QUANTIZED_SIMILARITIES_COUNT * templateDbCount * sizeof(unsigned int));
	memset(zeroMatrix, 0, QUANTIZED_SIMILARITIES_COUNT * templateDbCount * sizeof(unsigned int));
	CUDAArray<unsigned int> preBucketMatrix = CUDAArray<unsigned int>(zeroMatrix, QUANTIZED_SIMILARITIES_COUNT, templateDbCount);
	cudaMemcpyToSymbol(bucketMatrix, &preBucketMatrix, sizeof(CUDAArray<unsigned int>));
	cudaCheckError();
	generateBucketMatrix << <ceilMod(cylinderDbCount, THREADS_PER_BLOCK_MATRIX_GEN), THREADS_PER_BLOCK_MATRIX_GEN >> >(
		LUTSqrt, LUTAngles, LUTPopCountXor);
	cudaMemcpyFromSymbol(&preBucketMatrix, bucketMatrix, sizeof(CUDAArray<unsigned int>));
	printCUDAArray2D(preBucketMatrix);
	cudaCheckError();

	CUDAArray<float> similaritiesVector(templateDbCount, 1);
	computeLSS << <ceilMod(templateDbCount, THREADS_PER_BLOCK_LSS), THREADS_PER_BLOCK_LSS >> >
		(LUTTemplateDbLengths, LUTNumPairs, similaritiesVector);

	//LUTSqrt.Dispose();
	//LUTAngles.Dispose();
	//LUTNumPairs.Dispose();
	//bucketMatrix.Dispose();

	float* result = similaritiesVector.GetData();

	//similaritiesVector.Dispose();

	return result;
	//return nullptr;
}

int main()
{
	unsigned int cylinderCapacity = 1;

	unsigned int *cylinder0Values = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
	unsigned int *cylinder1Values = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));
	unsigned int *cylinder2Values = (unsigned int *)malloc(cylinderCapacity * sizeof(unsigned int));

	// Test 1
	//memset(cudaCylinder1, 255, cylinderCapacity * sizeof(unsigned int));
	//memset(cudaCylinder2, 255, cylinderCapacity * sizeof(unsigned int));


	// Test 2
	//srand((unsigned int)time(NULL));
	//for (unsigned int i = 0; i < cylinderCapacity; i++) {
	//	cudaCylinder1[i] = rand();
	//	cudaCylinder2[i] = rand();
	//}

	// Test 3 (only for cylinderCapacity == 1)

	cylinder0Values[0] = binToInt("00000000000000000000000000000000");
	cylinder1Values[0] = binToInt("11111111111111111100000000000000");
	cylinder2Values[0] = binToInt("11010001010100001100000000000000");

	Cylinder cylinder0 =
		Cylinder(cylinder0Values, cylinderCapacity, CUDART_PI_F / 6, sqrt((float)getOneBitsCountRaw(cylinder0Values, cylinderCapacity)), 0);
	Cylinder cylinder1_0 =
		Cylinder(cylinder1Values, cylinderCapacity, CUDART_PI_F / 4, sqrt((float)getOneBitsCountRaw(cylinder1Values, cylinderCapacity)), 0);
	Cylinder cylinder1_1 =
		Cylinder(cylinder1Values, cylinderCapacity, CUDART_PI_F / 4, sqrt((float)getOneBitsCountRaw(cylinder1Values, cylinderCapacity)), 1);
	Cylinder cylinder2_1 =
		Cylinder(cylinder2Values, cylinderCapacity, CUDART_PI_F / 3, sqrt((float)getOneBitsCountRaw(cylinder2Values, cylinderCapacity)), 1);
	Cylinder cylinder2_2 =
		Cylinder(cylinder2Values, cylinderCapacity, CUDART_PI_F / 3, sqrt((float)getOneBitsCountRaw(cylinder2Values, cylinderCapacity)), 2);

	Cylinder db[] = { cylinder1_0, cylinder1_1, cylinder2_1, cylinder2_2, cylinder2_2, cylinder2_2, cylinder2_2 };
	Cylinder query[] = { cylinder2_2 }; // Template index hopefully doesn't matter here
	unsigned int dbTemplateLengths[] = { 1, 2, 4 };
	unsigned int dbTemplateCount = sizeof(dbTemplateLengths) / sizeof(unsigned int);

	float *similarities = getBinTemplateSimilarities(
		query, sizeof(query) / sizeof(Cylinder),
		db, sizeof(db) / sizeof(Cylinder),
		dbTemplateLengths, dbTemplateCount);
	
	printf("Printing similarities:\n");
	for (unsigned int i = 0; i < dbTemplateCount; i++)
	{
		printf("%f%s", similarities[i], (i == dbTemplateCount - 1 ? "" : ", "));
	}
	printf("\n");

	// [end] Test 3

	free(cylinder0Values);
	free(cylinder1Values);
	free(cylinder2Values);

	return 0;
}
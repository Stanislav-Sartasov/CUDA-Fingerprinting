#ifndef CUDAFINGERPRINTING_BINTEMPLATECORRELATION
#define CUDAFINGERPRINTING_BINTEMPLATECORRELATION

#include "CylinderHelper.cuh"
#include "BinCorrelationHelper.cuh"

__device__ float getAngleDiff(float angle1, float angle2);

#define MAX_CYLINDERS_PER_TEMPLATE 256
#define CYLINDER_CELLS_COUNT 8 // Hopefully this define is not necessary (constant memory again)

#define DB_LENGTH 10000
#define MAX_QUERY_LENGTH 64

#define QUANTIZED_ANGLES_COUNT 256
#define QUANTIZED_SIMILARITIES_COUNT 64 // Basically buckets count

#define ANGLE_THRESHOLD 0.52359877f // == PI / 6

#define THREADS_PER_BLOCK_MATRIX_GEN 192
#define THREADS_PER_BLOCK_LSS 64

#define NUM_PAIRS_MIN 11
#define NUM_PAIRS_MAX 13
#define NUM_PAIRS_MU 30
#define NUM_PAIRS_TAU 0.4

extern "C"
{
	__declspec(dllexport) void initMCC(
		Cylinder *cylinderDb, unsigned int cylinderDbCount,
		unsigned int *templateDbLengths, unsigned int templateDbCount);

	__declspec(dllexport) float * processMCC(
		Cylinder *query, unsigned int queryLength,
		unsigned int cylinderDbCount, unsigned int templateDbCount);

	__declspec(dllexport) unsigned int checkValsFromTest(
		Cylinder *query, unsigned int queryLength,
		unsigned int cylinderDbCount, unsigned int templateDbCount);
}

void initMCC(
	Cylinder *cylinderDb, unsigned int cylinderDbCount,
	unsigned int *templateDbLengths, unsigned int templateDbCount);

float * processMCC(
	Cylinder *query, unsigned int queryLength,
	unsigned int cylinderDbCount, unsigned int templateDbCount);

#endif CUDAFINGERPRINTING_BINTEMPLATECORRELATION
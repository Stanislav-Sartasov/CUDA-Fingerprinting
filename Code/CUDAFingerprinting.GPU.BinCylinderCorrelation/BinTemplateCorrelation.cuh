#ifndef CUDAFINGERPRINTING_BINTEMPLATECORRELATION
#define CUDAFINGERPRINTING_BINTEMPLATECORRELATION

#include "CylinderHelper.cuh"
#include "BinCorrelationHelper.cuh"

#define MAX_CYLINDERS_PER_TEMPLATE 256
#define CYLINDER_CELLS_COUNT 8 // Hopefully this define is not necessary (constant memory again)

#define QUANTIZED_ANGLES_COUNT 256
#define QUANTIZED_SIMILARITIES_COUNT 64 // Basically buckets count

#define ANGLE_THRESHOLD 0.52359877f // == PI / 6

#define THREADS_PER_BLOCK_MATRIX_GEN 192
#define THREADS_PER_BLOCK_LSS 64

#define NUM_PAIRS_MIN 11
#define NUM_PAIRS_MAX 13
#define NUM_PAIRS_MU 30
#define NUM_PAIRS_TAU 0.4

float * getBinTemplateSimilarities(
	Cylinder *query, unsigned int queryLength,
	Cylinder *cylinderDb, unsigned int cylinderDbCount,
	unsigned int *templateDbLengths, unsigned int templateDbCount);

#endif CUDAFINGERPRINTING_BINTEMPLATECORRELATION
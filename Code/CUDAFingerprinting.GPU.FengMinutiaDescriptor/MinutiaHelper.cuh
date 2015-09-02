#ifndef CUDAFINGERPRINTING_MINUTIAHELPER
#define CUDAFINGERPRINTING_MINUTIAHELPER

#include "MinutiaStructs.cuh"
#include "math.h"
#include "constsmacros.h"

#ifndef M_PI
#define M_PI				3.14159265358979323846f
#endif

#define DESCRIPTOR_RADIUS	70 //TODO: count radius (width and heigth of picture)
#define COMPARE_RADIUS		5
#define FENG_CONSTANT		0.64f //0.8 * 0.8 
#define FENG_ANGLE			1.74532925f //100 deg
#define DESC_PER_BLOCK		8
#define DESC_BLOCK_SIZE		16
#define MAX_DESC_SIZE		128
#define FILENAME_LENGTH		5
#define FILEPATH_LENGTH		100

#define defaultDescriptorRow() blockIdx.y / DESC_PER_BLOCK
#define defaultDescriptorColumn() blockIdx.x / DESC_PER_BLOCK
#define defaultFinger() blockIdx.z

__host__ __device__ float sqrLength(Minutia m1, Minutia m2);

void fingersBaseRead(char *dbPath, int dbSize, int pitch, Minutia *mins, int *minutiaNum);

void fingerRead(char *filePath, Minutia *mins, int *minutiaNum);

__host__ __device__ void normalizeAngle(float *angle);

__device__ void cudaReductionSum(float* a, int i, int x);

__device__ void cudaReductionSum2D(float* a, int i, int j, int x, int y);

#endif
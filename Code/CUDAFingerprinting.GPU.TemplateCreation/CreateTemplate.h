#ifndef CUDAFINGERPRINTING_CREATETEMPLATE
#define CUDAFINGERPRINTING_CREATETEMPLATE
#include "math_constants.h"
struct Consts
{
	const char radius = 70;
	const char baseCuboid = 8;
	const char heightCuboid = 5;
	const unsigned int numberCell = baseCuboid * baseCuboid * heightCuboid;
	const float baseCell = (2.0 * radius) / baseCuboid;
	const float heightCell = (2 * CUDART_PI) / heightCuboid;
	const float sigmaLocation = 28.0 / 3;
	const float sigmaDirection = 2 * CUDART_PI / 9;
	const float sigmoidParametrPsi = 0.01;
	const char omega = 50;
	const char minNumberMinutiae = 2;
};

#define defaultX() threadIdx.x+1
#define defaultY() threadIdx.y+1
#define defaultZ() (blockIdx.y+1)*(threadIdx.z+1)
#define defaultMinutia() blockIdx.x

#define linearizationIndex() (defaultZ()-1)*constsGPU.baseCuboid*constsGPU.baseCuboid+(defaultY()-1)*constsGPU.baseCuboid+defaultX()-1
#endif
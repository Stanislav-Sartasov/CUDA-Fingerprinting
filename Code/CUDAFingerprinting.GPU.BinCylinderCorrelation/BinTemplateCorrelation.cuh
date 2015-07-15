#ifndef CUDAFINGERPRINTING_BINTEMPLATECORRELATION
#define CUDAFINGERPRINTING_BINTEMPLATECORRELATION

#include "CylinderHelper.cuh"

void getBinTemplateSimilarity(
	Cylinder *query, unsigned int queryLength,
	Cylinder *cylindersDb, unsigned int cylinderDbCount,
	unsigned int *templateDbLengths, unsigned int templateDbCount,
	float *similarityRates);

#endif CUDAFINGERPRINTING_BINTEMPLATECORRELATION
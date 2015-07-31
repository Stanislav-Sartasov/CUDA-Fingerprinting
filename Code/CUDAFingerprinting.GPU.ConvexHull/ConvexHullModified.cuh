#ifndef CUDAFINGERPRINTING_CONVEXHULLMODIFIED
#define CUDAFINGERPRINTING_CONVEXHULLMODIFIED

#include "ConvexHull.cuh"

extern "C"
{
	__declspec(dllexport) void initConvexHull(int givenFieldHeight, int givenFieldWidth, int givenMaxPointCount);
	__declspec(dllexport) bool* processConvexHull(Point* points, float omega, int actualPointCount);
	__declspec(dllexport) void terminateConvexHull();
}

void initConvexHull(int givenFieldHeight, int givenFieldWidth, int givenMaxPointCount);
bool* processConvexHull(Point* points, float omega, int actualPointCount);
void terminateConvexHull();

#endif CUDAFINGERPRINTING_CONVEXHULLMODIFIED
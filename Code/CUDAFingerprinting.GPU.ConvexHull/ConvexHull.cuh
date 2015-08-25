#ifndef CUDAFINGERPRINTING_CONVEXHULL
#define CUDAFINGERPRINTING_CONVEXHULL

#include "VectorHelper.cuh"

// Basically Graham scan
void getConvexHull(Point* points, int pointsLength, Point* hull, int *hullLength);

bool** getFieldFilling(int rows, int columns, Point* hull, int hullLength);
__global__ void getFieldFillingParallel(int rows, int columns, Point* hull, int hullLength, bool* field, int pitch);

#endif
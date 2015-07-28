#ifndef CUDAFINGERPRINTING_CONVEXHULLMODIFIED
#define CUDAFINGERPRINTING_CONVEXHULLMODIFIED

#include "ConvexHull.cuh"

Point* extendHull(Point* hull, int hullLength, float omega);

bool** getRoundFieldFilling(
	int rows, int columns, float omega, Point* hull, int hullLength, Point* extendedHull, int extendedHullLength);

#endif CUDAFINGERPRINTING_CONVEXHULLMODIFIED
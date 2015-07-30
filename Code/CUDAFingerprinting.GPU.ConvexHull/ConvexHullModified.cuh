#ifndef CUDAFINGERPRINTING_CONVEXHULLMODIFIED
#define CUDAFINGERPRINTING_CONVEXHULLMODIFIED

#include "ConvexHull.cuh"

extern "C"
{
	__declspec(dllexport) Point* extendHull(Point* hull, int hullLength, float omega);

	__declspec(dllexport) bool** getRoundFieldFilling(
		int rows, int columns, float omega, Point* hull, int hullLength, Point* extendedHull, int extendedHullLength);

	// This could be not tested (image saved in test using C# saver, 
	__declspec(dllexport) void printHullMathCoords(bool* field, char *filename);
}

Point* extendHull(Point* hull, int hullLength, float omega);

bool** getRoundFieldFilling(
	int rows, int columns, float omega, Point* hull, int hullLength, Point* extendedHull, int extendedHullLength);

void printHullMathCoords(bool* field, char *filename);

#endif CUDAFINGERPRINTING_CONVEXHULLMODIFIED
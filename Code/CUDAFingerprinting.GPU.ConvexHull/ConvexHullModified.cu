#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "constsmacros.h"
#include "ImageLoading.cuh"
#include "ConvexHullModified.cuh"

#define DEBUG
//#define NONPARALLEL

#define MAX_FILE_NAME_LENGTH 1000
#define MAX_FILE_LINE_LENGTH 100

int maxPointCount;
int pointCount;
int fieldWidth;
int fieldHeight;

#ifdef DEBUG
#define cudaCheckError() {\
	cudaError_t e = cudaGetLastError();\
	if (e != cudaSuccess) {\
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));\
		exit(0);\
	}\
}
#else
#define cudaCheckError() ;
#endif

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BLOCK_DIM 32

Point* hull;
// Is it possible to store this in constant memory? 
// As for now it's not passed properly as a parameter 
// (global var usage bad practice & impossible from another file)
//__constant__ Point d_extendedHull[pointCount * 2];
bool* extendedRoundedField;
size_t pitch;
bool* d_extendedRoundedField;

// Modified hull code

Point* extendHull(Point* hull, int hullLength, float omega)
{
	Point* extendedHull = (Point *)malloc(hullLength * 2 * sizeof(Point));

	Point fst, snd;
	int i = 0;
	while (i < hullLength)
	{
		fst = hull[i];
		snd = hull[(i + 1) % hullLength];

		Point diff = difference(snd, fst);

		Point orthogonalDiff = Point(diff.y, -diff.x);
		float orthogonalDiffNorm = norm(orthogonalDiff);

		Point moveVector = Point(
			orthogonalDiff.x / orthogonalDiffNorm * omega,
			orthogonalDiff.y / orthogonalDiffNorm * omega);

		extendedHull[i * 2] = Point(fst.x + moveVector.x, fst.y + moveVector.y);
		extendedHull[i * 2 + 1] = Point(snd.x + moveVector.x, snd.y + moveVector.y);

		i++;
	}

	return extendedHull;
}

void getRoundFieldFilling(
	int rows, int columns, float omega, Point* hull, int hullLength, Point* extendedHull, int extendedHullLength, bool* field)
{
	for (int i = 1; i < extendedHullLength; i += 2)
	{
		Point hullPoint = hull[((i + 1) / 2) % hullLength]; // All these modulos for border cases (i + 1 = extendedHull.Count)

		int jMin = MAX((int)round(hullPoint.x - omega), 0);
		int jMax = MIN((int)round(hullPoint.x + omega), rows);
		int kMin = MAX((int)round(hullPoint.y - omega), 0);
		int kMax = MIN((int)round(hullPoint.y + omega), columns);
		for (int j = jMin; j < jMax; j++)
		{
			for (int k = kMin; k < kMax; k++)
			{
				Point curPoint = Point((float)j, (float)k);
				if (pointDistance(curPoint, hullPoint) < omega)
				{
					field[j * columns + k] = true;
				}
			}
		}
	}
}

// [end] Modified hull code
// Printing hull code (file & console)

#ifdef DEBUG

void printHullMathCoords(bool** field, char *filename)
{
	int* linearizedField = (int*)malloc(fieldHeight * fieldWidth * sizeof(int));
	for (int i = 0; i < fieldHeight; i++)
	{
		for (int j = 0; j < fieldWidth; j++)
		{
			//linearizedField[i * fieldWidth + j] = field[i][j] == true ? 255 : 0;

			// BTW why is the pic gray (not black) if all the pixels are 0?
			// AFAIK GPU ImageLoading doesn't reflect image upside down, 
			// thus simple transpose doesn't suffice, vertical reflection needed
			linearizedField[(fieldHeight - 1 - j) * fieldWidth + i] = field[i][j] == true ? 255 : 0;
		}
	}

	saveBmp(filename, linearizedField, fieldWidth, fieldHeight);

	free(linearizedField);
}

void printHullMathCoords(bool* field, char *filename)
{
	int* intField = (int*)malloc(fieldHeight * fieldWidth * sizeof(int));

	for (int i = 0; i < fieldHeight * fieldWidth; i++)
	{
		//intField[i] = field[i] == true ? 255 : 0;
		intField[
			(fieldHeight - 1 - (i % fieldWidth)) * fieldWidth +
				(i / fieldWidth)] =
				field[i] == true ? 255 : 0;
	}

	saveBmp(filename, intField, fieldWidth, fieldHeight);

	free(intField);
}

void printHull(Point* hull, int hullLength)
{
	printf("Printing hull\n");
	for (int i = 0; i < hullLength; i++)
	{
		printf("%f, %f\n", hull[i].x, hull[i].y);
	}
	printf("[end] Printing hull\n");
}

#endif

// [end] Printing hull code (file & console)

void initConvexHull(int givenFieldHeight, int givenFieldWidth, int givenMaxPointCount)
{	
	fieldHeight = givenFieldHeight;
	fieldWidth = givenFieldWidth;
	maxPointCount = givenMaxPointCount;

	hull = (Point*)malloc(maxPointCount * sizeof(Point)); // maximum hull length

	extendedRoundedField = (bool *)malloc(fieldHeight * fieldWidth * sizeof(bool));

#ifndef NONPARALLEL
	cudaMallocPitch((void**)&d_extendedRoundedField, &pitch, fieldWidth, fieldHeight);
#endif
}

bool* processConvexHull(Point* points, float omega, int actualPointCount)
{
	pointCount = actualPointCount;

#ifndef NONPARALLEL
	int maxExtendedHullLength = pointCount * 2;
	Point* d_extendedHull;
	cudaMalloc((void **)&d_extendedHull, sizeof(Point)* maxExtendedHullLength);
#endif

	// TODO Switch to CUDA events & move to main()
	//clock_t start = clock();

	int hullLength; // var used to bound "actual" hull length
	getConvexHull(points, pointCount, hull, &hullLength);

	Point* extendedHull = extendHull(hull, hullLength, omega);
	int extendedHullLength = hullLength * 2;

#ifdef DEBUG
	// TODO
	// Change pointCount into smth else (extern maybe)
	// Modify printing such that hull & extended hull can be distinguished
	printHull(hull, hullLength);
	printHull(extendedHull, extendedHullLength);
#endif

#ifdef NONPARALLEL
	bool** field = getFieldFilling(fieldHeight, fieldWidth, hull, hullLength);
	bool** extendedField = getFieldFilling(fieldHeight, fieldWidth, extendedHull, extendedHullLength);
	// For now nonparallel version without rounded field (due to not matching signatures),
	// legacy code, somewhat how it should look like
	//bool** extendedRoundedField = getRoundFieldFilling(
	//	fieldHeight, fieldWidth, omega, hull, hullLength, extendedHull, extendedHullLength);
#else
	cudaMemcpy(d_extendedHull, extendedHull, sizeof(Point)* extendedHullLength, cudaMemcpyHostToDevice);
	//cudaMemcpyToSymbol(d_extendedHull, extendedHull, sizeof(Point)* extendedHullLength);
	cudaCheckError();

	dim3 gridDim = dim3(ceilMod(fieldWidth, BLOCK_DIM), ceilMod(fieldWidth, BLOCK_DIM));
	dim3 blockDim = dim3(BLOCK_DIM, BLOCK_DIM);
	getFieldFillingParallel << < gridDim, blockDim >> > (
		fieldHeight, fieldWidth, d_extendedHull, extendedHullLength, d_extendedRoundedField, pitch);
	cudaDeviceSynchronize();
	cudaCheckError();

	cudaMemcpy2D(
		extendedRoundedField, fieldWidth * sizeof(bool), d_extendedRoundedField, pitch,
		fieldWidth, fieldHeight, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaCheckError();

#ifdef DEBUG
#ifndef NONPARALLEL
	// Still getting parallel in order not to perform additional cudaMemcpy, just for debug
	bool** field = getFieldFilling(fieldHeight, fieldWidth, hull, hullLength);
	printHullMathCoords(field, "ConvexHull.jpg");
	free(field);

	printHullMathCoords(extendedRoundedField, "ConvexHullExtended.jpg");
#endif
#endif

	getRoundFieldFilling(
		fieldHeight, fieldWidth, omega, hull, hullLength, extendedHull, extendedHullLength, extendedRoundedField);
#endif

	//clock_t end = clock();
	//printf("TIME: %ld\n", end - start);

#ifdef DEBUG
#ifdef NONPARALLEL
	printHullMathCoords(field, "ConvexHull.jpg");
	printHullMathCoords(extendedRoundedField, "ConvexHullExtended.jpg");
#else
	printHullMathCoords(extendedRoundedField, "ConvexHullExtendedRounded.jpg");
#endif
#endif

	// Freeing to in termination in order not to create additional global vars for nonparallel
	// (which would be initialized separately)
	// (just no need to speed up nonparallel version)
#ifdef NONPARALLEL
	free(field);
	free(extendedField);
#else
	cudaFree(d_extendedHull);
#endif

	return extendedRoundedField;
	//bool* res = (bool*)malloc(fieldHeight * fieldWidth * sizeof(bool));
	//return res;
}

void terminateConvexHull()
{
	free(hull);
#ifndef NONPARALLEL
	//cudaFree(d_extendedHull);
	free(extendedRoundedField);
	cudaFree(d_extendedRoundedField);
#endif
}

// input file should be in format:
// <num of points>
// x, y
// x, y
// ...
// Could be generated at random.org/integer-sets/
int parsePoints(char* path, Point** points)
{
	FILE *file = fopen(path, "r");
	if (!file)
	{
		printf("Point DB open error\n");
		exit(EXIT_FAILURE);
	}

	char* line = new char[MAX_FILE_LINE_LENGTH];

	fgets(line, MAX_FILE_LINE_LENGTH, file);
	pointCount = strtoul(line, NULL, 10);

	*points = (Point*)malloc(pointCount * sizeof(Point));

	for (int i = 0; i < pointCount; i++)
	{
		fgets(line, MAX_FILE_LINE_LENGTH, file);

		char* xStr = strtok(line, ", ");
		float x = strtof(xStr, NULL);
		char* yStr = strtok(NULL, " ");
		float y = strtof(yStr, NULL);

		(*points)[i] = Point(x, y);
	}

	return pointCount;
}


/*int main()
{
	char pathPointDb[MAX_FILE_NAME_LENGTH] = "C:\\Users\\resaglow\\convex_hull_db.txt";
	Point* points = nullptr;
	int actualPointCount = parsePoints(pathPointDb, &points);

	float omega = 40;

	initConvexHull(1100, 1100, 500);
	processConvexHull(points, omega, actualPointCount);
	terminateConvexHull();

	return 0;
}*/
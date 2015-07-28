#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ImageLoading.cuh"
#include "ConvexHullModified.cuh"

#define DEBUG

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

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

bool** getRoundFieldFilling(
	int rows, int columns, float omega, Point* hull, int hullLength, Point* extendedHull, int extendedHullLength)
{
	bool** field = getFieldFilling(rows, columns, extendedHull, extendedHullLength);

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
					field[j][k] = true;
				}
			}
		}
	}

	return field;
}

void printHullMathCoords(bool** field, char *filename)
{
	int* linearizedField = (int*)malloc(TEST_FIELD_HEIGHT * TEST_FIELD_WIDTH * sizeof(int));
	for (int i = 0; i < TEST_FIELD_HEIGHT; i++)
	{
		for (int j = 0; j < TEST_FIELD_WIDTH; j++)
		{
			// BTW why is the pic gray (not black) if all the pixels are 0?
			// AFAIK GPU ImageLoading doesn't invert image upside down, thus simple transpose doesn't suffice
			// No idea how to interpret this geometrically, yet it works
			linearizedField[(TEST_FIELD_HEIGHT - 1 - j) * TEST_FIELD_WIDTH + i] = field[i][j] == true ? 255 : 0;
		}
	}

	saveBmp(filename, linearizedField, TEST_FIELD_WIDTH, TEST_FIELD_HEIGHT);

	free(linearizedField);
}

int main()
{
	Point points[TEST_POINT_COUNT] =
	{
		Point(0, 100),
		Point(200, 0),
		Point(400, 200),
		Point(800, 300),
		Point(600, 600),
		Point(300, 700),
		Point(200, 600),
		Point(100, 900)
	};

	float omega = 40;

	clock_t start = clock();

	Point* hull = (Point*)malloc(TEST_POINT_COUNT * sizeof(Point)); // maximum hull length
	int hullLength; // var used to bound "actual" hull length

	getConvexHull(points, TEST_POINT_COUNT, hull, &hullLength);

	Point* extendedHull = extendHull(hull, hullLength, omega);
	int extendedHullLength = hullLength * 2;

#ifdef DEBUG
	printf("Printing hull\n");
	for (int i = 0; i < hullLength; i++)
	{
		printf("%f, %f\n", hull[i].x, hull[i].y);
	}
	printf("[end] Printing hull\n");

	printf("\n");

	printf("Printing extended hull\n");
	for (int i = 0; i < extendedHullLength; i++)
	{
		printf("%f, %f\n", extendedHull[i].x, extendedHull[i].y);
	}
	printf("[end] Printing exnteded hull\n");
#endif

#ifdef DEBUG
	bool** field = getFieldFilling(TEST_FIELD_HEIGHT, TEST_FIELD_WIDTH, hull, hullLength);
	bool** extendedField = getFieldFilling(TEST_FIELD_HEIGHT, TEST_FIELD_WIDTH, extendedHull, extendedHullLength);
#endif
	bool** extendedRoundedField = getRoundFieldFilling(
		TEST_FIELD_HEIGHT, TEST_FIELD_WIDTH, omega, hull, hullLength, extendedHull, extendedHullLength);

#ifdef DEBUG
	printHullMathCoords(field, "ConvexHull.jpg");
	printHullMathCoords(extendedField, "ConvexHullExtended.jpg");
	printHullMathCoords(extendedRoundedField, "ConvexHullExtendedRounded.jpg");
#endif


	free(hull);
#ifdef DEBUG
	free(field);
	free(extendedField);
#endif
	free(extendedRoundedField);

	clock_t end = clock();
	printf("TIME: %ld\n", end - start);

	return 0;
}
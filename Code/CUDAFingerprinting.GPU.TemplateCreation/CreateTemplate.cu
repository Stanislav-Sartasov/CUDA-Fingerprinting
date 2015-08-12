
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "constsmacros.h"
#include <stdio.h>
#include "CUDAArray.cuh"
#include "CylinderHelper.cuh"
#include "VectorHelper.cuh"
#include "ConvexHull.cu"
#include "ConvexHullModified.cu"
#include "math_functions.hpp"
#include "math.h"
#include "math_functions_dbl_ptx3.h"
struct Minutia
{
	float angle;
	int x;
	int y;
};

void createTemplate(Minutia* minutiae, int lenght, Cylinder* cylinders, int* cylindersLenght)
{
	Point* points = (Point*)malloc(lenght * sizeof(Point));
	CUDAArray<Minutia> cudaMinutiae = CUDAArray<Minutia>(minutiae, lenght, 1);
	CUDAArray<Point> cudaPoints = CUDAArray<Point>(points, lenght, 1);
	free(points);

	getPoints << <1, lenght >> >(cudaMinutiae, cudaPoints, lenght);
	cudaMinutiae.Dispose();
	int* hullLenght;
	Point* hull;

	getConvexHull(cudaPoints.GetData, lenght, hull, hullLenght);
	cudaPoints.Dispose();
	Point* exdHull = extendHull(hull, *hullLenght, 50);//50 - omega 
	free(hull);
	int* extLenght;
	*extLenght = *hullLenght * 2;
	free(hullLenght);


}

__global__ void getValidMinutias(CUDAArray<Minutia> minutiae, CUDAArray<bool> validMinutiae)
{
	int colomn = defaultColumn();

	if (blockIdx.x < minutiae.Width || colomn < minutiae.Width)
	{
		int newX = minutiae.At[0, blockIdx.x].x - minutiae.At[0, colomn].x;
		int newY = minutiae.At[0, blockIdx.x].y - minutiae.At[0, colomn].y;
		validMinutiae.SetAt(0, colomn, sqrt((float)(newX*newX + newY*newY)) < 70 + 3 * 28 ? true : false);// radius + 3*sigma
	}
}

__global__ void getPoints(CUDAArray<Minutia> minutiae, CUDAArray<Point> points, int lenght)
{
	int column = defaultColumn();

	if (column < lenght)
	{
		points.SetAt(0, column, Point(minutiae.At(0, column).x, minutiae.At(0, column).y));
	}
}

int main()
{

}


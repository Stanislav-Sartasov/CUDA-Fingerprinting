#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include "VectorHelper.cuh"

__device__ __host__ float pointDistance(Point A, Point B)
{
	float diffX = B.x - A.x;
	float diffY = B.y - A.y;
	return sqrt(diffX * diffX + diffY * diffY);
}

float norm(Point v)
{
	return sqrt(v.x * v.x + v.y * v.y);
}

__device__ __host__ float vectorProduct(Point v1, Point v2)
{
	return v1.x * v2.y - v1.y * v2.x;
}

__device__ __host__ Point difference(Point v1, Point v2)
{
	Point vDiff;
	vDiff.x = v1.x - v2.x;
	vDiff.y = v1.y - v2.y;
	return vDiff;
}

__device__ __host__ float rotate(Point A, Point B, Point C)
{
	return vectorProduct(difference(B, A), difference(C, B));
}

__device__ __host__ bool intersect(Point A, Point B, Point C, Point D)
{
	// <= in the 1st case and < in the second are appropriate for the specific use of this helper (localization problem)
	return rotate(A, B, C) * rotate(A, B, D) <= 0 && rotate(C, D, A) * rotate(C, D, B) < 0;
}
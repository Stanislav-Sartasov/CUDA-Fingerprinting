#ifndef CUDAFINGERPRINTING_VECTORHELPER
#define CUDAFINGERPRINTING_VECTORHELPER

struct Point
{
	float x;
	float y;

	__device__ __host__ Point() {}
	__device__ __host__ Point(float x, float y) : x(x), y(y) {}
};

__device__ __host__  float pointDistance(Point A, Point B);
float norm(Point v);

// Vector product of 2 vectors (only z coordinate, given vectors are supposed to be arranged on a plane)
__device__ __host__ float vectorProduct(Point v1, Point v2);

__device__ __host__ Point difference(Point v1, Point v2);

// Helper function for 3 points 
// A, B, C -> going from A to B, where is C, to the left or to the right?
// > 0 - left (positive rotation)
// = 0 - all 3 points are collinear
// < 0 - right
__device__ __host__ float rotate(Point A, Point B, Point C);

// Segment intersection for localization problem
__device__ __host__ bool intersect(Point A, Point B, Point C, Point D);

#endif
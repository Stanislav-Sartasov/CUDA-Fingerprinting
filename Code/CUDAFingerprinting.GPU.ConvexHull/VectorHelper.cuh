#ifndef CUDAFINGERPRINTING_VECTORHELPER
#define CUDAFINGERPRINTING_VECTORHELPER

struct Minutia
{
	float angle;
	int x;
	int y;
};

struct Point
{
	float x;
	float y;

	Point() {}
	Point(float x, float y) : x(x), y(y) {}
};

float pointDistance(Point A, Point B);
float norm(Point v);

// Vector product of 2 vectors (only z coordinate, given vectors are supposed to be arranged on a plane)
float vectorProduct(Point v1, Point v2);

Point difference(Point v1, Point v2);

// Helper function for 3 points 
// A, B, C -> going from A to B, where is C, to the left or to the right?
// > 0 - left (positive rotation)
// = 0 - all 3 points are collinear
// < 0 - right
float rotate(Point A, Point B, Point C);

// Segment intersection for localization problem
bool intersect(Point A, Point B, Point C, Point D);

#endif
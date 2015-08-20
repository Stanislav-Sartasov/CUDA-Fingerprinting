#include "MinutiaHelper.cuh"

__device__ float sqrLength(Minutia m1, Minutia m2)
{
	return (float)((m1.x - m2.x)*(m1.x - m2.x) + (m1.y - m2.y)*(m1.y - m2.y));
}

__device__ float normalizeAngle(float angle)
{
	float res = angle - (float)(floor(angle / (2 * M_PI)) * 2 * M_PI);
}
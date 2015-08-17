#include "MinutiaHelper.cuh"

__device__ float sqrLength(Minutia m1, Minutia m2)
{
	return (float)((m1.x - m2.x)*(m1.x - m2.x) + (m1.y - m2.y)*(m1.y - m2.y));
}
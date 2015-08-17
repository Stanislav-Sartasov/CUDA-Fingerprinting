#include "cuda_runtime.h"
#include "MinutiaHelper.cuh"
#include "math.h"
#include "Constants.cuh"

//#include "device_launch_parameters.h"


__global__ void transformate(CUDAArray<Minutia> src, Minutia center, CUDAArray<Minutia> dst)
{
	int row = defaultRow();
	int column = defaultColumn();

	float angle = center.angle - src.At(row, column).angle;
	float cosAngle = cos(angle);
	float sinAngle = sin(angle);
	float dx = src.At(row, column).x - center.x;
	float dy = src.At(row, column).y - center.y;

	int x = (int)round(dx * cosAngle + dy * sinAngle) + center.x;
	int y = (int)round(-dx * sinAngle + dy * cosAngle) + center.y;

	Minutia temp(src.At(row, column).angle + angle, x, y);

	dst.SetAt(row, column, temp);
}

__global__ void compare(Descriptor desc1, Descriptor desc2, int* m, int* M, int height, int width)
{
	extern __shared__ bool cache[]; //first half for m, second for M

	int row = defaultRow();
	int column = defaultColumn();
	int cacheIndex1 = row*blockDim.y + column;
	int halfCacheSize = blockDim.x * blockDim.y / 2;
	int cacheIndex2 = cacheIndex1 + halfCacheSize;

	float eps = 0.1;

	if ((sqrLength(desc1.minutias[row], desc2.minutias[column]) < COMPARE_RADIUS*COMPARE_RADIUS)
		&& ((desc1.minutias[row].angle - desc2.minutias[column].angle) < eps))
	{
		cache[cacheIndex1] = 1;
		cache[cacheIndex2] = 1;
	}
	else
	{
		cache[cacheIndex1] = 0;
		if ((sqrLength(desc1.minutias[row], desc2.center) < FENG_CONSTANT * DESCRIPTOR_RADIUS * DESCRIPTOR_RADIUS) &&
			(desc1.minutias[row].x >= 0 && desc1.minutias[row].x < width
			&& desc1.minutias[row].y >= 0 && desc1.minutias[row].y < height))
		{
			cache[cacheIndex2] = 1;
		}
		else
		{
			cache[cacheIndex2] = 0;
		}
	}

	__syncthreads();

	int i = halfCacheSize / 4;
	while (i != 0)
	{
		if (cacheIndex1 < i)
		{
			cache[cacheIndex1] += cache[cacheIndex1 + i];
		}
		else if (cacheIndex1 < 2*i)
		{
			cache[cacheIndex2 - i] += cache[cacheIndex2];
		}

		__syncthreads();
		i /= 2;
	}

	*m = cache[0];
	*M = cache[halfCacheSize];
}

#ifdef DEBUG
int main()
{

	return 0;
}
#endif
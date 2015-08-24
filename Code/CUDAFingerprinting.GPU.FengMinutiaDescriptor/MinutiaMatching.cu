#include "cuda_runtime.h"
#include "MinutiaHelper.cuh"

#include <math.h>
#include <stdlib.h>

__global__ void makeNormalSize(float*** s, float*** sn) ///block 8*8
{
	__shared__ float sum[DESC_PER_BLOCK][DESC_PER_BLOCK];
	
	int row = defaultRow();
	int column = defaultColumn();
	int k = defaultFinger();

	sum[threadIdx.y][threadIdx.x] = s[k][row][column];

	cudaReductionSum2D((float*)sum, DESC_PER_BLOCK, DESC_PER_BLOCK, threadIdx.y, threadIdx.x);

	if ((threadIdx.y == 0) && (threadIdx.x == 0))
	{
		sn[k][blockIdx.y][blockIdx.x] = sum[0][0];
	}
}

__global__ void sumRow(float** rowSum, float*** s, float*** temp) ///block 1*1  grid 1000*128*128
{
	int row = defaultRow();
	int column = defaultColumn();
	int k = blockIdx.z;

	temp[k][row][column] = s[k][row][column];

	int i = MAX_DESC_SIZE / 2;
	while (i != 0)
	{
		if (row < i)
		{
			atomicAdd(&temp[k][row][column], temp[k][row + i][column]);
		}

		i /= 2;
	}

	if (row == 0)
	{
		rowSum[k][column] = s[k][0][column];
	}
}

__global__ void sumColumn(float** columnSum, float*** s, float*** temp) 
{
	int row = defaultRow();
	int column = defaultColumn();
	int k = blockIdx.z;

	temp[k][row][column] = s[k][row][column];

	int i = MAX_DESC_SIZE / 2;
	while (i != 0)
	{
		if (column < i)
		{
			atomicAdd(&temp[k][row][column], temp[k][row][column + i]);
		}

		i /= 2;
	}

	if (column == 0)
	{
		columnSum[k][column] = s[k][0][column];
	}
}

__global__ void normalize(float*** s, Minutia* input, Minutia** current, float** rowSum, float** columnSum,
	int n, int* m) 
{
	int row = defaultRow();
	int column = defaultColumn();
	int k = blockIdx.z;

	if (abs(input[row].angle - current[k][column].angle) < FENG_ANGLE)
	{
		s[k][row][column] = s[k][row][column] * (n + m[k] - 1) / (rowSum[k][row] + columnSum[k][column] - s[k][row][column]);
	}
	else
	{
		s[k][row][column] = 0;
	}
}

__global__ void makeTuple(float** s, Tuple** tuples) ///s - 3D array
{
	int row = defaultRow();
	int column = defaultColumn();

	tuples[row][column].value = s[row][column];
}

int comparator(const void *a, const void *b)
{
	Tuple x = *(Tuple *)a;
	Tuple y = *(Tuple *)b;

	return y.value - x.value >= 0 ? -1 : 1;
}

__device__ float length(Minutia m1, Minutia m2)
{
	return (float)sqrt(sqrLength(m1, m2));
}

__device__ bool isMatchable(Minutia m1, Minutia m2, Minutia kernel1, Minutia kernel2)
{
	bool isOnSameDistance, isClose, isOnSameDirection;
	float epsAngle = 0.3F;
	float a1, a2, dist1, dist2, chordk, chordm;

	dist1 = length(m1, kernel1);
	dist2 = length(m2, kernel2);
	isOnSameDistance = abs(dist1 - dist2) < COMPARE_RADIUS;

	a1 = normalizeAngle(kernel1.angle - kernel2.angle);
	a2 = normalizeAngle(m1.angle - m2.angle);
	isOnSameDirection = abs(a1 - a2) < epsAngle;

	chordk = (float)sin(abs(a1 / 2)) * dist1 * 2;
	Minutia tempm;
	tempm.angle = 0.0F;
	tempm.x = m2.x + (kernel1.x - kernel2.x);
	tempm.y = m2.y + (kernel1.y - kernel2.y);
	chordm = length(m1, tempm);
	isClose = abs(chordk - chordm) < COMPARE_RADIUS;

	return isOnSameDistance && isClose && isOnSameDirection;
}


__global__ void matchMinutias(Tuple** tuples, int n, Minutia* input, Minutia** current, int** res)  ///block 1024 tuples must be sorted!
{
	__shared__ bool flag1[MAX_DESC_SIZE];
	__shared__ bool flag2[MAX_DESC_SIZE];

	int row = defaultRow(); //top - 100?
	int column = defaultColumn(); //1024 - blocksize
	int k = defaultFinger(); //1000

	if ((k == 0) && (row == 0) && (column < MAX_DESC_SIZE))
	{
		flag1[column] = false;
		flag2[column] = false;
	}

	int i0 = tuples[k][row].idx1;
	int j0 = tuples[k][row].idx2;

	flag1[i0] = true;
	flag2[j0] = true;

	int t = column;
	int i, j;
	int temp = 0;

	while (t < n)
	{
		i = tuples[k][t].idx1;
		j = tuples[k][t].idx2;

		if (!flag1[i] && !flag2[j] && isMatchable(input[i], current[k][j], input[i0], current[k][j0]))
		{
			atomicAdd(&temp, 1);

			flag1[i] = true;
			flag2[j] = true;
		}

		t += column;
	}

	res[k][row] = temp;
}
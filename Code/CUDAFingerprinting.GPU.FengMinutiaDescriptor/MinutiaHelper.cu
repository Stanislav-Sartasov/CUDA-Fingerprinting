#include <stdio.h>
#include <stdlib.h>
#include <string.h> 

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaHelper.cuh"

__device__ float sqrLength(Minutia m1, Minutia m2)
{
	return (float)((m1.x - m2.x)*(m1.x - m2.x) + (m1.y - m2.y)*(m1.y - m2.y));
}

__device__ float normalizeAngle(float angle)
{
	float res = angle - (float)(floor(angle / (2 * M_PI)) * 2 * M_PI);
}

__global__ void fingerRead (char *dbPath, int dbSize, Minutia **mins, int *minutiaNum)
{
	FILE *finger;
	int pathLength = 100;
	char filePath[1] = "";
	int num = blockIdx.x;
	char fileNum[FILENAME_LENGTH];
	int i;

	if (num >= dbSize)
	{
		return;
	}

	/*itoa(num, fileNum, 10);
	strncat(filePath, dbPath, FILENAME_LENGTH);
	strncat(filePath, fileNum, FILENAME_LENGTH);
	strncat(filePath, ".txt", FILENAME_LENGTH);

	finger = fopen(dbPath, "r");

	fscanf(finger, "%d", minutiaNum[num]);
	for (i = 0; i < *minutiaNum; i++)
	{
		fscanf(finger, "%d %d %f", mins[num][i].x, mins[num][i].y, mins[num][i].angle);
	}

	fclose(finger);
*/}
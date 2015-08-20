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
	angle -= (float)(floor(angle / (2 * M_PI)) * 2 * M_PI);

	return angle;
}

__global__ void fingerRead(char *dbPath, int dbSize, Minutia **mins, int *minutiaNum)
{
	FILE *finger;
	char filePath[FILEPATH_LENGTH];
	char fileNum[FILENAME_LENGTH];
	int i, j;

	for (i = 0; i < dbSize; i++)
	{
		filePath[0] = '\0';
		itoa(i, fileNum, 10);
		strncat(filePath, dbPath, FILEPATH_LENGTH);
		strncat(filePath, "\\", FILENAME_LENGTH);
		strncat(filePath, fileNum, FILENAME_LENGTH);
		strncat(filePath, ".txt", FILENAME_LENGTH);

		finger = fopen(filePath, "r");

		fscanf(finger, "%d", &(minutiaNum[i]));
		for (j = 0; j < minutiaNum[i]; j++)
		{
			fscanf(finger, "%d %d %f", &(mins[i][j].x), &(mins[i][j].y), &(mins[i][j].angle));
		}

		fclose(finger);
		/*
		for (j = 0; j < minutiaNum[i]; j++)
		{
			printf("%d %d %f\n", mins[i][j].x, mins[i][j].y, mins[i][j].angle);
		}
		printf("__\n");*/
	}
}

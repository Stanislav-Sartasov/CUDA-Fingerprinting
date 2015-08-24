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

__device__ void normalizeAngle(float *angle)
{
	*angle -= (float)(floor(*angle / (2 * M_PI)) * 2 * M_PI);
}

void fingersBaseRead(char *dbPath, int dbSize, int pitch, Minutia *mins, int *minutiaNum)
{
	FILE *finger;
	char filePath[FILEPATH_LENGTH];
	char fileNum[FILENAME_LENGTH];
	int i, j = 0;

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
			fscanf(finger, "%d %d %f", &(mins[i*pitch + j].x), &(mins[i*pitch + j].y), &(mins[i*pitch + j].angle));
		}
		fclose(finger);
		/*
		for (j = 0; j < minutiaNum[i]; j++)
		{
			printf("%d %d %f\n", mins[i*pitch + j].x, mins[i*pitch + j].y, mins[i*pitch + j].angle);
		}
		printf("__%d\n",i);*/
	}
}

void fingerRead(char *filePath, Minutia *mins, int *minutiaNum)
{
	FILE *finger;
	int i;
	finger = fopen(filePath, "r");
	fscanf(finger, "%d", minutiaNum);
	for (i = 0; i < *minutiaNum; i++)
	{
		fscanf(finger, "%d %d %f", &(mins[i].x), &(mins[i].y), &(mins[i].angle));
	}
	/*
	for (int j = 0; j < *minutiaNum; j++)
	{
		printf("%d %d %f\n", mins[j].x, mins[j].y, mins[j].angle);
	}
	*/
	fclose(finger);
}

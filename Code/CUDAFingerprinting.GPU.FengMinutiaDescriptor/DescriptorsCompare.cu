#include "cuda_runtime.h"
#include "MinutiaHelper.cuh"
#include "DescriptorBuilder.cuh"
#include <stdio.h>
#include <stdlib.h>

//#include "device_launch_parameters.h"

__device__ void matchingPoints(Descriptor* desc1, Descriptor* desc2, int* m, int* M, int width, int height)
{
	int i;
	float angle = (*desc2).center.angle - (*desc1).center.angle;
	float cosAngle = cos(angle);
	float sinAngle = sin(angle);

	*m = 0;
	*M = 0;
	
	for (i = 0; i < (*desc1).length; i++)
	{
		int dx = (*desc1).minutias[i].x - (*desc1).center.x;
		int dy = (*desc1).minutias[i].y - (*desc1).center.y;

		int x = (int)round(dx * cosAngle + dy * sinAngle) + (*desc2).center.x;
		int y = (int)round(-dx * sinAngle + dy * cosAngle) + (*desc2).center.y;

		Minutia min;
		min.angle = (*desc1).minutias[i].angle + angle;
		normalizeAngle(&(min.angle));
		min.x = x;
		min.y = y;

		float eps = 0.1;
		
		int j = 0;
		bool isExist = false;
		while ((j < (*desc2).length) && !isExist)
		{
			if ((sqrLength((*desc2).minutias[j], min) < COMPARE_RADIUS*COMPARE_RADIUS)
				&& (abs((*desc2).minutias[j].angle - min.angle) < eps))
			{
				isExist = true;
			}
			j++;
		}

		if (isExist)
		{
			++*m; 
			++*M;
		}
		else
		{
			if ((sqrLength(min, (*desc2).center) < FENG_CONSTANT * DESCRIPTOR_RADIUS * DESCRIPTOR_RADIUS) &&
				(min.x >= 0 && min.x < width && min.y >= 0 && min.y < height))
			{
				++*M;
			}
		}
	}
}

__global__ void compareDescriptors(Descriptor* input, Descriptor* current, int height, int width, int pitch, float* s,
	int inputNum, int* currentNum)
{
	int x = defaultColumn();
	int k = blockIdx.z;
	int y = defaultRow();

	if (x < inputNum && y < currentNum[k])
	{
		int m1, M1, m2, M2;

		matchingPoints(&input[x], &current[k*pitch + y], &m1, &M1, width, height);

		matchingPoints(&current[k*pitch + y], &input[x], &m2, &M2, width, height);


		s[k*MAX_DESC_SIZE*MAX_DESC_SIZE + x*MAX_DESC_SIZE + y] = (1.0 + m1) * (1.0 + m2) / (1.0 + M1) / (1.0 + M2);
	}
}

void dbCompare() //in progress
{
	cudaSetDevice(0);
	int i, j;
	int sizeOfMin = sizeof(Minutia);
	int sizeOfDesc = sizeof(Descriptor);
	int sizeOfInt = sizeof(int);
	int minPitch = MAX_DESC_SIZE * sizeOfMin;
	int height = 364;
	int width = 265;

	printf("1000x128 blocks with 128 threads each\n");
	printf("1000 lists of minutia database into base of lists of descriptors\n");

	/////////work with input finger
	int fingerMinutiaNum;
	int *dev_fingerMinutiaNum;
	cudaMalloc((void**)&dev_fingerMinutiaNum, sizeOfInt);
	char *fingerPath = "D:\\inputFinger.txt";

	Minutia *fingerMins = (Minutia*)malloc(minPitch);

	Minutia *dev_fingerMins;
	cudaMalloc((void**)&dev_fingerMins, minPitch);
	fingerRead(fingerPath, fingerMins, &fingerMinutiaNum);
	cudaMemcpy(dev_fingerMins, fingerMins, minPitch, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_fingerMinutiaNum, &fingerMinutiaNum, sizeOfInt, cudaMemcpyHostToDevice);

	Descriptor *dev_fingerDesc;  //TODO: try constant memory
	cudaMalloc((void**)&dev_fingerDesc, MAX_DESC_SIZE * sizeOfDesc);

	buildDescriptors << <dim3(1, MAX_DESC_SIZE), MAX_DESC_SIZE >> >(dev_fingerMins, 1, dev_fingerMinutiaNum, dev_fingerDesc, 1);
	/////////end of work with input finger

	/////////  work with fingers base
	int dbSize = 1000;
	int *dbMinutiaNum = (int*)malloc(dbSize*sizeOfInt);
	int *dev_dbMinutiaNum;
	cudaMalloc((void**)&dev_dbMinutiaNum, sizeOfInt*dbSize);

	char *dbPath = "D:\\FingersBase";

	Minutia *dbMins = (Minutia*)malloc(dbSize * minPitch);

	Minutia *dev_dbMins;
	cudaMalloc((void**)&dev_dbMins, dbSize * minPitch);

	fingersBaseRead(dbPath, dbSize, MAX_DESC_SIZE, dbMins, dbMinutiaNum); // done
	cudaMemcpy(dev_dbMins, dbMins, minPitch*dbSize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dbMinutiaNum, dbMinutiaNum, dbSize * sizeOfInt, cudaMemcpyHostToDevice);


	Descriptor *dev_dbDesc;
	cudaMalloc((void**)&dev_dbDesc, MAX_DESC_SIZE*dbSize*sizeOfDesc);


	buildDescriptors << <dim3(dbSize, MAX_DESC_SIZE), MAX_DESC_SIZE >> >(dev_dbMins, MAX_DESC_SIZE, dev_dbMinutiaNum, dev_dbDesc, dbSize);

	///////end of work with fingers base

	///////compare descriptors


	float *s;
	float* cpu_s = (float*)malloc(MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float));

	cudaMalloc((void**)&s, MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float));

	compareDescriptors << < 4 * 4 * dbSize,
		dim3(32, 32) >> > (
		dev_fingerDesc, dev_dbDesc, height, width, MAX_DESC_SIZE, s, fingerMinutiaNum, dev_dbMinutiaNum);

	cudaMemcpy(cpu_s, s, MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float), cudaMemcpyDeviceToHost);

	int topSize = 128;
	float *top;
	cudaMalloc((void**)&top, topSize*dbSize*sizeof(float));
	float* cpu_top = (float*)malloc(topSize*dbSize*sizeof(float));

	cudaMemcpy(cpu_top, top, topSize*dbSize*sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(s);
	free(cpu_s);
	cudaFree(top);
	free(cpu_top);
	free(fingerMins);
	cudaFree(dev_fingerMinutiaNum);
	cudaFree(dev_fingerMins);
	cudaFree(dev_fingerDesc);

	free(dbMinutiaNum);
	free(dbMins);
	cudaFree(dev_dbMinutiaNum);
	cudaFree(dev_dbMins);
	cudaFree(dev_dbDesc);
}

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaHelper.cuh"
#include "DescriptorBuilder.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main()
{
	cudaSetDevice(0);
	int i, j;
	int sizeOfMin = sizeof(Minutia);
	int sizeOfDesc = sizeof(Descriptor);
	int sizeOfInt = sizeof(int);
	int minPitch = MAX_DESC_SIZE * sizeOfMin;

	printf("1000x128 blocks with 128 threads each\n");
	printf("1000 lists of minutia database into base of lists of descriptors\n");
	for (int k = 0; k < 10; k++)
	{
		printf("\nrun num %d:\n", k);
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

		Descriptor *dev_fingerDesc;
		cudaMalloc((void**)&dev_fingerDesc, sizeOfDesc);

		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);

		buildDescriptors <<<dim3(1, MAX_DESC_SIZE), MAX_DESC_SIZE >>>(dev_fingerMins, 1, dev_fingerMinutiaNum, dev_fingerDesc, 1);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		float et;
		cudaEventElapsedTime(&et, start, stop);
		printf("for finger   %3.1fms \n", et);


		free(fingerMins);
		cudaFree(dev_fingerMinutiaNum);
		cudaFree(dev_fingerMins);
		cudaFree(dev_fingerDesc);

		/////////end of work with input finger






		/////////work with fingers base
		int dbSize = 1000;
		int *dbMinutiaNum = (int*)malloc(dbSize*sizeOfInt);
		int *dev_dbMinutiaNum;
		cudaMalloc((void**)&dev_dbMinutiaNum, dbSize*sizeOfInt);
		char *dbPath = "D:\\FingersBase";


		Minutia *dbMins = (Minutia*)malloc(dbSize * minPitch);

		Minutia *dev_dbMins;
			cudaMalloc((void**)&dev_dbMins, dbSize * minPitch);

		fingersBaseRead(dbPath, dbSize, MAX_DESC_SIZE, dbMins, dbMinutiaNum); // done
		cudaMemcpy(dev_dbMinutiaNum, dbMinutiaNum, dbSize * sizeOfInt, cudaMemcpyHostToDevice);
		cudaMemcpy(dev_dbMins, dbMins, minPitch*dbSize, cudaMemcpyHostToDevice);

		Descriptor *dev_dbDesc;
		cudaMalloc((void**)&dev_dbDesc, MAX_DESC_SIZE*dbSize*sizeOfDesc);

		cudaEventRecord(start, 0);

		buildDescriptors <<<dim3(dbSize, MAX_DESC_SIZE), MAX_DESC_SIZE >>>(dev_dbMins, MAX_DESC_SIZE, dev_dbMinutiaNum, dev_dbDesc, dbSize);

		cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&et, start, stop);
		printf("for base     %3.1fms\n", et);

		free(dbMinutiaNum);
		free(dbMins);
		cudaFree(dev_dbMinutiaNum);
		cudaFree(dev_dbMins);
		cudaFree(dev_dbDesc);
		///////end of work with fingers base



	}
	printf("\n");



	/*
	int count;
	cudaGetDeviceCount(&count);

	for (int i = 0; i < count; i++)
	{
		cudaGetDeviceProperties(&prop, i);
		printf("name %s\n", prop.name);
		printf("overlap %d\n", prop.deviceOverlap);
		printf("multiproc %d\n", prop.multiProcessorCount);
		printf("threads %d\n", prop.maxThreadsPerBlock);
		printf("threads per dem %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("clockRate %d\n", prop.clockRate);
		printf("asyncEngines %d\n", prop.asyncEngineCount);
		printf("multyKernels %d\n", prop.concurrentKernels);
		printf("maxGridSize %d\n", prop.maxGridSize);
		
	}*/
}
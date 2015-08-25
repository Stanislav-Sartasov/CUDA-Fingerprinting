
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaHelper.cuh"
#include "DescriptorBuilder.cuh"
#include "DescriptorsCompare.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void k(int *p)
{
	printf("nums\n");
	for (int i = 262; i < 272; i++)
	{
		printf("adr %d\n", &(p[i]));
	}
}

__global__ void l(Descriptor *p)
{
	printf("descs\n");
	for (int i = 0; i < 10; i++)
	{
		printf("adr %d\n", &(p[i]));
	}
}


int main()
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

	Descriptor *dev_fingerDesc;
	cudaMalloc((void**)&dev_fingerDesc, MAX_DESC_SIZE * sizeOfDesc);


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

	cudaEventRecord(start, 0);

	buildDescriptors <<<dim3(dbSize, MAX_DESC_SIZE), MAX_DESC_SIZE >>>(dev_dbMins, MAX_DESC_SIZE, dev_dbMinutiaNum, dev_dbDesc, dbSize);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);
	printf("for base     %3.1fms\n", et);

		
	///////end of work with fingers base

	///////compare descriptors
	
	Descriptor *temp0, *temp1;
	cudaMalloc((void**)&temp0, MAX_DESC_SIZE*dbSize*sizeOfDesc);
	cudaMalloc((void**)&temp1, MAX_DESC_SIZE*dbSize*sizeOfDesc);

	float *s;
	cudaMalloc((void**)&s, MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float));
	printf(".1fms\n", et);
	//compareDescriptors << < dim3(MAX_DESC_SIZE / DESC_PER_BLOCK, MAX_DESC_SIZE / DESC_PER_BLOCK, dbSize),
		//dim3(DESC_BLOCK_SIZE, DESC_BLOCK_SIZE, 2) >>> (
		//dev_fingerDesc, dev_dbDesc, temp0, temp1, s, height, width, MAX_DESC_SIZE);
	printf("3.1fms\n");
	///////
	cudaFree(s);
	cudaFree(temp0);
	cudaFree(temp1);

	free(fingerMins);
	cudaFree(dev_fingerMinutiaNum);
	cudaFree(dev_fingerMins);
	cudaFree(dev_fingerDesc);
	
	free(dbMinutiaNum);
	free(dbMins);
	cudaFree(dev_dbMinutiaNum);
	cudaFree(dev_dbMins);
	cudaFree(dev_dbDesc);
	
	//int eddsf = 0;
	//scanf("%d", eddsf);



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
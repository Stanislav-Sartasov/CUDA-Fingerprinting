#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MinutiaHelper.cuh"
#include "MinutiaMatching.cuh"
#include "DescriptorBuilder.cuh"
#include "DescriptorsCompare.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void sh(float *s)
{
	for (int i = 0; i < 1000; i++)
		for (int j = 0; j < 128; j++)
			for (int k = 0; k < 128; k++)
			{
				s[128 * 128 * i + 128 * j + k] = (128 * j + k);
			}
}

__global__ void sortParts(float *s, int partSize)
{

}

__global__ void t(int *p)
{
	printf("matrx\n");
	for (int i = 0; i < 1; i++)
	{
		for (int j = 0; j < 100; j++)
		{
			printf("%d\n", p[j]);
		}
	}
}

__global__ void k(Descriptor *p, Descriptor *b)
{
	printf("\n");
	printf("%d %d %f\n", p[0].center.x, p[0].center.y, p[0].center.angle);
	printf("\n");
	printf("%d %d %f\n", b[0].center.x, b[0].center.y, b[0].center.angle);
	printf("\n");
}

__global__ void l(Descriptor *p)
{
	printf("descs\n");
	for (int i = 0; i < 10; i++)
	{
		printf("len: %d\n", (p[i].length));
		printf("center: %d %d %f\n", p[i].center.x, p[i].center.y, p[i].center.angle);
		for (int j = 0; j < p[i].length; j++)
		{
			printf("%d %d %f\n", p[i].minutias[j].x, p[i].minutias[j].y, p[i].minutias[j].angle);
		}
	}
}
//__constant__ Descriptor fingerDesc[MAX_DESC_SIZE];

__global__ void count(int* s_m1, int* s_M1, int* s_m2, int* s_M2, float* s)
{
	for (int k = 0; k < 1; ++k)
	{
		for (int i = 0; i < 30; ++i)
		{
			for (int j = 0; j < 30; ++j)
			{
				s[i * MAX_DESC_SIZE + j] = (1.0 + s_m1[i * MAX_DESC_SIZE + j]) * (1.0 + s_m2[i * MAX_DESC_SIZE + j]) /
					(1.0 + s_M1[i * MAX_DESC_SIZE + j]) / (1.0 + s_M2[i * MAX_DESC_SIZE + j]);
				//printf("%d %d\n", s_m1[i * MAX_DESC_SIZE + j], s_M1[i * MAX_DESC_SIZE + j]);
			}
			//printf("\n");
		}
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

	//buildFingerDescriptors(fingerMins, &fingerMinutiaNum, fingerDesc);

	//cudaMemcpyToSymbol(dev_fingerMins, fingerMins, minPitch);
	cudaMemcpy(dev_fingerMinutiaNum, &fingerMinutiaNum, sizeOfInt, cudaMemcpyHostToDevice);

	Descriptor *dev_fingerDesc;  //TODO: try constant memory
	cudaMalloc((void**)&dev_fingerDesc, MAX_DESC_SIZE * sizeOfDesc);
//	cudaMemcpyToSymbol(dev_fingerDesc, fingerDesc, MAX_DESC_SIZE * sizeOfDesc);


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
	//l <<<1, 1>>> (dev_fingerDesc);
	//k << <1, 1 >> > (dev_fingerDesc, dev_dbDesc);
	///////compare descriptors
	

	float *s;
	float* cpu_s = (float*)malloc(MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float));

	cudaMalloc((void**)&s, MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float));

	printf(".1fms\n", et);
	cudaEventRecord(start, 0);
	/*compareDescriptors << < 4 * 4 ,
		dim3(32, 32) >> > (
		dev_fingerDesc, dev_dbDesc, height, width, MAX_DESC_SIZE, s, fingerMinutiaNum, dev_dbMinutiaNum);*/
	//l << <1, 1 >> > (dev_dbDesc);
	sh <<<1, 1>>>(s);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);
	printf("for cmp   %3.1fms \n", et);

	//l << <1, 1 >> > (dev_fingerDesc);
	//t << < 1, 1 >>> (s_m1);
	printf("3.1fms\n");
	///////
	cudaMemcpy(cpu_s, s, MAX_DESC_SIZE*MAX_DESC_SIZE*dbSize*sizeof(float), cudaMemcpyDeviceToHost);
	
	FILE* f;
	f = fopen("file.txt", "w");
	for (int k = 100; k < 101; ++k)
	{
		for (int i = 0; i < 30; ++i)
		{
			for (int j = 0; j < 30; ++j)
			{
				fprintf(f, "%f ", cpu_s[i * MAX_DESC_SIZE + j]);
			}
			fprintf(f, "\n");
		}
	}
	

	int topSize = 32;
	float *top;
	cudaMalloc((void**)&top, topSize*dbSize*sizeof(float));
	float* cpu_top = (float*)malloc(topSize*dbSize*sizeof(float));

	
	cudaEventRecord(start, 0);

	topElements<<<dim3(dbSize,2), 512>>>(s, MAX_DESC_SIZE*MAX_DESC_SIZE, MAX_DESC_SIZE, top, topSize);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&et, start, stop);
	printf("for topSearch   %3.1fms \n", et);

	cudaMemcpy(cpu_top, top, topSize*dbSize*sizeof(float), cudaMemcpyDeviceToHost);
	FILE* ff;
	ff = fopen("D:\\file1.txt", "w");
	for (int k = 0; k < 100; ++k)
	{
		for (int j = 0; j < 32; ++j)
		{
			fprintf(ff, "%f ", cpu_top[k * topSize + j]);
		}
		fprintf(ff, "\n");
	}
	printf("%d", 128 * 128 * sizeof(float) / 1024);
	cudaFree(s);
	free(cpu_s);
	cudaFree(top);
	free(cpu_top);
	free(fingerMins);
//	cudaFree(dev_fingerMinutiaNum);
	//cudaFree(dev_fingerMins);
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
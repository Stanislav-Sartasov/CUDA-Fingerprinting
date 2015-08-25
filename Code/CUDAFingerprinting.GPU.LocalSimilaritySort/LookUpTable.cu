#include "LookUpTable.cuh"

#include <algorithm>
#include <fstream>

const short LUT_NP::MAX_CYLINDERS = 256;
const short LUT_NP::MIN_NP = 11;
const short LUT_NP::MAX_NP = 13;
const float LUT_NP::TAU = 0.4;
const float LUT_NP::MU = 30.0;

//template<class T>
//bool LUT<T>::canBeRead(const char fileName[]) {
//	std::ifstream fin(fileName);
//	bool openStatus = (fin != NULL);
//	fin.close();
//	return openStatus;
//}

void LUT_NP::fill(short upperBound) {
	size = upperBound < MAX_CYLINDERS ? upperBound : MAX_CYLINDERS;
	short* cpuPtr = new short[size];
	for (int i = 0; i < size; ++i)
		cpuPtr[i] = MIN_NP + (short)lrint((MAX_NP - MIN_NP) / (1 + exp(-TAU * (i - MU))));

	cudaError_t cudaStatus = cudaMalloc(&cudaPtr, size * sizeof(short));
	cudaStatus = cudaMemcpy(cudaPtr, cpuPtr, size * sizeof(short), cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short>();
	cudaBindTexture(0, &texLUTSource, cudaPtr, &channelDesc, size * sizeof(short));

	delete[] cpuPtr;
}

int LUT_NP::read(const char fileName[], size_t limit) {
	std::ifstream fin(fileName);
	if (!fin) {
		fin.close();
		return -1;
	}

	fin >> size;

	size = size < limit ? size : limit;
	
	short* cpuPtr = new short[size];
	for (int i = 0; i < size; ++i)
		fin >> cpuPtr[i];

	cudaError_t cudaStatus = cudaMalloc(&cudaPtr, size * sizeof(short));
	cudaStatus = cudaMemcpy(cudaPtr, cpuPtr, size * sizeof(short), cudaMemcpyHostToDevice);

	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<short>();
	cudaBindTexture(0, &texLUTSource, cudaPtr, &channelDesc, size * sizeof(short));

	delete[] cpuPtr;

	fin.close();
	return 0;
}

__device__ short LUT_NP::operator[](size_t pos) const {
	return tex1Dfetch(texLUTSource, pos);
}

void LUT_NP::free() {
	cudaUnbindTexture(texLUTSource);
	cudaFree(cudaPtr);
}

void LUT_OR::fill(size_t templatesNumber, short templateSizes[]) {
	size = templatesNumber;

	int* cpuPtr = new int[size + 1];

	cpuPtr[0] = 0;
	for (int i = 0; i < templatesNumber; ++i)
		cpuPtr[i + 1] = templateSizes[i] + cpuPtr[i];

	databaseHeight = cpuPtr[size];

	cudaError_t cudaStatus = cudaMalloc(&cudaPtr, (size + 1) * sizeof(int));
	cudaStatus = cudaMemcpy(cudaPtr, cpuPtr, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);

	delete[] cpuPtr;
}

int LUT_OR::read(const char fileName[]) {
	std::ifstream fin(fileName);

	if (!fin) {
		fin.close();
		return -1;
	}

	fin >> size;

	int* cpuPtr = new int[size + 1];

	cpuPtr[0] = 0;
	for (int i = 0; i < size; ++i) {
		fin >> cpuPtr[i + 1];
		cpuPtr[i + 1] += cpuPtr[i];
	}

	databaseHeight = cpuPtr[size];

	cudaError_t cudaStatus = cudaMalloc(&cudaPtr, (size + 1) * sizeof(int));
	cudaStatus = cudaMemcpy(cudaPtr, cpuPtr, (size + 1) * sizeof(int), cudaMemcpyHostToDevice);

	delete[] cpuPtr;

	fin.close();
	return 0;
}

__device__ int LUT_OR::operator[](size_t pos) const {
	return cudaPtr[pos];
}

void LUT_OR::free() {
	cudaFree(cudaPtr);
}
#include "LookUpTable.cuh"

#include <algorithm>
#include <fstream>

const short LUT_NP::MAX_CYLINDERS = 256;
const short LUT_NP::MIN_NP = 11;
const short LUT_NP::MAX_NP = 13;
const float LUT_NP::TAU = 0.4;
const float LUT_NP::MU = 30.0;

template<class T>
bool LUT<T>::canBeRead(const char fileName[]) {
	return fopen(fileName, "r") != NULL;
}

void LUT_NP::fill(int upperBound) {
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

void LUT_NP::read(const char fileName[], size_t limit) {
	std::ifstream fin(fileName);
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
}

__device__ short LUT_NP::operator[](int pos) {
	return tex1Dfetch(texLUTSource, pos);
}

void LUT_NP::free() {
	cudaUnbindTexture(texLUTSource);
	cudaFree(cudaPtr);
}

void LUT_C::fill(short templateSizes[], size_t templatesNumber) {
	size = 0;
	for (int i = 0; i < templatesNumber; ++i)
		size += templateSizes[i];

	int* cpuPtr = new int[size];
	int k = 0;
	for (int i = 0; i < templatesNumber; ++i)
	for (short j = 0; j < templateSizes[i]; ++j)
		cpuPtr[k++] = i;

	cudaError_t cudaStatus = cudaMalloc(&cudaPtr, size * sizeof(int));
	cudaStatus = cudaMemcpy(cudaPtr, cpuPtr, size * sizeof(int), cudaMemcpyHostToDevice);

	delete[] cpuPtr;
}

void LUT_C::read(const char fileName[]) {
	std::ifstream fin(fileName);
	fin >> size;

	int* cpuPtr = new int[size];

	for (int i = 0; i < size; ++i)
		fin >> cpuPtr[i];

	cudaError_t cudaStatus = cudaMalloc(&cudaPtr, size * sizeof(int));
	cudaStatus = cudaMemcpy(cudaPtr, cpuPtr, size * sizeof(int), cudaMemcpyHostToDevice);

	delete[] cpuPtr;

	fin.close();
}

__device__ int LUT_C::operator[](int pos) {
	return cudaPtr[pos];
}

void LUT_C::free() {
	cudaFree(cudaPtr);
}
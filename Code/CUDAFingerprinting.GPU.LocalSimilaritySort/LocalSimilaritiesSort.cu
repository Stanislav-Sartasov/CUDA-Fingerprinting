// Large-scale fingerprint identification on GPU
// Raffaele Cappelli, Matteo Ferrara, Davide Maltoni, 2015

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDAArray.cuh"
#include "LookUpTable.cuh"
#include "LocalSimilaritiesSort.cuh"
#include <cmath>

#ifdef _DEBUG
#include <algorithm>
#include <cstdio>
#include <fstream>
#endif

const char INPUT_FILE_TEMPLATE_TAGS[] = "C:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprintig.GPU.LocalSimilaritySort\\tags.txt";
const char INPUT_FILE_NP_VALS[] = "C:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprintig.GPU.LocalSimilaritySort\\np.txt";
const short quantLevels = 64;


template<class T>
inline __device__  T min(T left, T right) {
	return left < right ? left : right;
}


// GPU signatures

__global__ void quantify(CUDAArray<float> src, CUDAArray<short> dst, LUT_OR templateOriginRow);
__global__ void quantify(CUDAArray<short> src, CUDAArray<short> dst, LUT_OR templateOriginRow);

__global__ void cudaComputeGlobalScores(float* globalScores, CUDAArray<short> bucketMatrix, short queryTemplateSize, short* templateSizes, LUT_NP numberOfValuesToPick);


// CPU signatures

template<class T>
T* linearizeDoublePointer(T** matrix, size_t wOffset, size_t height, size_t width);

template<class T>
CUDAArray<short> constructBuckets(T* similaritiesDatabase, short width, const LUT_OR &templateOriginRow);

void computeGlobalScoresFromBuckets(float* globalScores, CUDAArray<short> bucketMatrix, short* templateSizes, short queryTemplateSize, const LUT_NP &numberOfValuesToPick);


// External Functions

void getGlobalScoresShort(float* globalScores, short* similaritiesDatabase, int templatesNumber, short* templateSizes, short queryTemplateSize) {

	// Set up Look-up tables
	LUT_OR templateOriginRow;
	LUT_NP numberOfValuesToPick;

	//if (templateOriginRow.read(INPUT_FILE_TEMPLATE_TAGS) == -1)
	templateOriginRow.fill(templatesNumber, templateSizes);

	//if (numberOfValuesToPick.read(INPUT_FILE_NP_VALS) == -1)
	numberOfValuesToPick.fill();

	CUDAArray<short> bucketMatrix = constructBuckets(similaritiesDatabase, queryTemplateSize, templateOriginRow);
	
	computeGlobalScoresFromBuckets(globalScores, bucketMatrix, templateSizes, queryTemplateSize, numberOfValuesToPick);

	templateOriginRow.free();
	numberOfValuesToPick.free();
}

void getGlobalScoresFloat(float* globalScores, float* similaritiesDatabase, int templatesNumber, short* templateSizes, short queryTemplateSize) {

	// Set up Look-up tables
	LUT_OR templateOriginRow;
	LUT_NP numberOfValuesToPick;

	if (templateOriginRow.read(INPUT_FILE_TEMPLATE_TAGS) == -1)
		templateOriginRow.fill(templatesNumber, templateSizes);

	if (numberOfValuesToPick.read(INPUT_FILE_NP_VALS) == -1)
		numberOfValuesToPick.fill();

	CUDAArray<short> bucketMatrix = constructBuckets(similaritiesDatabase, queryTemplateSize, templateOriginRow);

	computeGlobalScoresFromBuckets(globalScores, bucketMatrix, templateSizes, queryTemplateSize, numberOfValuesToPick);

	templateOriginRow.free();
	numberOfValuesToPick.free();
}


// Leading Functions

template<class T>
CUDAArray<short> constructBuckets(T* similaritiesDatabase, short width, const LUT_OR &templateOriginRow) {
	cudaError_t cudaStatus;

	CUDAArray<short> bucketMatrix(quantLevels, templateOriginRow.getLength());

	// Copy database to GPU

	CUDAArray<T> gpuDB(similaritiesDatabase, width, templateOriginRow.databaseHeight);

	dim3 blockSize(defaultThreadCount, defaultThreadCount);
	dim3 gridSize(templateOriginRow.getLength());
	quantify<<<gridSize, blockSize>>>(gpuDB, bucketMatrix, templateOriginRow);

	cudaStatus = cudaGetLastError();
	// Should this be deleted?
	cudaStatus = cudaDeviceSynchronize();

	gpuDB.Dispose();

	return bucketMatrix;
}

void computeGlobalScoresFromBuckets(float* globalScores, CUDAArray<short> bucketMatrix, short* templateSizes, short queryTemplateSize, const LUT_NP &numberOfValuesToPick) {
	cudaError_t cudaStatus;

	short* gpuTemplateSizes = 0;
	cudaStatus = cudaMalloc(&gpuTemplateSizes, bucketMatrix.Height * sizeof(short));
	cudaStatus = cudaMemcpy(gpuTemplateSizes, templateSizes, bucketMatrix.Height * sizeof(short), cudaMemcpyHostToDevice);

	float* gpuGlobalScores = 0;
	cudaStatus = cudaMalloc(&gpuGlobalScores, bucketMatrix.Height * sizeof(float));

	dim3 blockSize = dim3(bucketMatrix.Width, defaultThreadCount * defaultThreadCount / bucketMatrix.Width);
	dim3 gridSize = ceilMod(bucketMatrix.Height, blockSize.x);

	cudaComputeGlobalScores <<<gridSize, blockSize, bucketMatrix.Width * bucketMatrix.Width * sizeof(short)>>>
		(gpuGlobalScores, bucketMatrix, queryTemplateSize, gpuTemplateSizes, numberOfValuesToPick);
	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(globalScores, gpuGlobalScores, bucketMatrix.Height * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(gpuGlobalScores);
	cudaFree(gpuTemplateSizes);
}


// CPU Implementaion

template<class T>
T* linearizeDoublePointer(T** matrix, size_t wOffset, size_t height, size_t width) {
	T* lin = new T[height*width];
	size_t bytesInRow = width * sizeof(T);
	for (size_t i = 0; i < height; ++i)
		memcpy(lin + i * width, matrix[i] + wOffset, bytesInRow);
	return lin;
}


// GPU Implemetation

__global__ void quantify(CUDAArray<float> src, CUDAArray<short> dst, LUT_OR templateOriginRow) {

	// allow only two threads address to the templateOriginRow
	__shared__ size_t yOffset[2];
	if ((threadIdx.x <= 1) && (threadIdx.y == 0))
		yOffset[threadIdx.x] = templateOriginRow[blockIdx.x + threadIdx.x];
	__syncthreads();
	
	__shared__ int subBucketMatrix[quantLevels];
	if (threadIdx.y < quantLevels / blockDim.x) 
		subBucketMatrix[blockDim.x * threadIdx.y + threadIdx.x] = 0;
	__syncthreads();

	size_t x;	//	database column
	size_t y = yOffset[0] + threadIdx.y;	//	database row
	
	// traverse down
	do {
		x = threadIdx.x;

		// traverse right
		do {
			if ((x < src.Width) && (y < yOffset[1])) {
				int level = (int)(quantLevels * src.At(y, x));
				level = min(level, 63);
				atomicAdd(subBucketMatrix + level, 1);
			}
			x += blockDim.x;
			__syncthreads();
		} while (x - threadIdx.x < src.Width);

		y += blockDim.y;
	} while (y - threadIdx.y < yOffset[1]);
	
	// coalesced copy to global memory
	if (threadIdx.y < quantLevels / blockDim.x) {
		short index = blockDim.x * threadIdx.y + threadIdx.x;
		dst.SetAt(blockIdx.x, index, subBucketMatrix[index]);
	}
	
}

__global__ void quantify(CUDAArray<short> src, CUDAArray<short> dst, LUT_OR templateOriginRow) {

	// allow only two threads address to the templateOriginRow
	__shared__ size_t yOffset[2];
	if ((threadIdx.x <= 1) && (threadIdx.y == 0))
		yOffset[threadIdx.x] = templateOriginRow[blockIdx.x + threadIdx.x];
	__syncthreads();

	__shared__ int subBucketMatrix[quantLevels];
	if (threadIdx.y < quantLevels / blockDim.x)
		subBucketMatrix[blockDim.x * threadIdx.y + threadIdx.x] = 0;
	__syncthreads();

	size_t x;	//	database column
	size_t y = yOffset[0] + threadIdx.y;	//	database row

	// traverse down
	do {
		x = threadIdx.x;

		// traverse right
		do {
			if ((x < src.Width) && (y < yOffset[1]))
				atomicAdd(subBucketMatrix + src.At(y, x), 1);
			x += blockDim.x;
			__syncthreads();
		} while (x - threadIdx.x < src.Width);

		y += blockDim.y;
	} while (y - threadIdx.y < yOffset[1]);

	// coalesced copy to global memory
	if (threadIdx.y < quantLevels / blockDim.x) {
		short index = blockDim.x * threadIdx.y + threadIdx.x;
		dst.SetAt(blockIdx.x, index, subBucketMatrix[index]);
	}
}

__global__ void cudaComputeGlobalScores(float* globalScores, CUDAArray<short> bucketMatrix, short queryTemplateSize, short* templateSizes, LUT_NP numberOfValuesToPick) {

	// Subcopy bucket matrix to shared memory

	size_t hOffset = blockDim.x * blockIdx.x;
	extern __shared__ short subBucketMatrix[];
	size_t rowsNumber = min(blockDim.x, bucketMatrix.Height - hOffset);

	for (int i = 0; i < rowsNumber; ++i)
		subBucketMatrix[quantLevels* i + threadIdx.x] = (short)bucketMatrix.At(hOffset + i, threadIdx.x);
	__syncthreads();


	// Count sum of the lowest values

	if (threadIdx.x < rowsNumber) {
		size_t currentTemplate = defaultColumn();
		const short np = numberOfValuesToPick[min(templateSizes[currentTemplate], queryTemplateSize)];
		short currentLevel = 0,
			canYetTake = np,
			sumOfLowest = 0;
		while ((currentLevel < bucketMatrix.Width) && (canYetTake > 0)) {
			short takeNow = min(subBucketMatrix[threadIdx.x * quantLevels + currentLevel], canYetTake);
			sumOfLowest += takeNow * currentLevel++;
			canYetTake -= takeNow;
		}
		globalScores[currentTemplate] = 1.0f - (float)sumOfLowest / (np * bucketMatrix.Width);
	}

	__syncthreads();

}

int main() {

	/*int templatesNumber = 180;
	short templateSizes[180];
	int width = 0;
	for (int i = 0; i < templatesNumber; ++i) {
	templateSizes[i] = 60 + rand() % 40;
	width += templateSizes[i];
	}
	short queryTemplateSize = 60 + rand() % 40;
	float** DB = new float*[queryTemplateSize];
	for (int i = 0; i < queryTemplateSize; ++i) {
	DB[i] = new float[width];
	for (int j = 0; j < width; ++j)
	DB[i][j] = (rand() % 256) / 256.0;
	}

	// auto linDB = linearizeDoublePointer(DB, 0, queryTemplateSize, width);
	*/

	std::ifstream fin_database("C:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprinting.GPU.LocalSimilaritySort\\database.txt");
	std::ifstream fin_tempalteSizes("C:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprinting.GPU.LocalSimilaritySort\\templateSizes.txt");

	int templatesNumber;
	short *templateSizes;
	int height, queryTemplateSize;
	float *linDB;

	fin_tempalteSizes >> templatesNumber;

	templateSizes = new short[templatesNumber];

	for (int i = 0; i < templatesNumber; ++i)
		fin_tempalteSizes >> templateSizes[i];

	fin_database >> height >> queryTemplateSize;

	linDB = new float[height * queryTemplateSize];

	for (int i = 0; i < height; ++i) {
		int hOffset = i * queryTemplateSize;
		for (int j = 0; j < queryTemplateSize; ++j)
			fin_database >> linDB[hOffset + j];
	}

	auto error = cudaSetDevice(0);

	float *scores = new float[templatesNumber];
	getGlobalScoresFloat(scores, linDB, templatesNumber, templateSizes, queryTemplateSize);
	for (int i = 0; i < templatesNumber; ++i)
		printf("%f\n", scores[i]);

	cudaDeviceReset();
	/*for (int i = 0; i < queryTemplateSize; ++i)
	delete[] DB[i];*/
	delete[] templateSizes;
	delete[] linDB;
	delete[] scores;

	return 0;
}
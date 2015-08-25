// Large - scale fingerprint identification on GPU
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

__global__ void quantify(CUDAArray<float> src, CUDAArray<int> dst, LUT_C templateTags);
__global__ void quantify(CUDAArray<short> src, CUDAArray<int> dst, LUT_C templateTags);

__global__ void cudaComputeGlobalScores(float* globalScores, CUDAArray<int> bucketMatrix, size_t subBMWidth, short queryTemplateSize, short* templateSizes, LUT_NP numberOfValuesToPick);


// CPU signatures

//template<class T>
//T maxElement(T *base, size_t length);

void spreadNumbers(short* src, int* dst, size_t srcLength);

template<class T>
size_t getPitch(size_t width);

template<class T>
T* linearizeDoublePointer(T** matrix, size_t wOffset, size_t height, size_t width);

template<class T>
CUDAArray<int> constructBuckets(T* similaritiesDatabase, int templatesNumber, const LUT_C &templateTags, short queryTemplateSize);

void computeGlobalScores(CUDAArray<int> bucketMatrix, short* templateSizes, short queryTemplateSize, const LUT_NP &numberOfValuesToPick, float cpuGlobalScores[]);


// General functions

void getGlobalScoresShort(short* similaritiesDatabase, int templatesNumber, short* templateSizes, short queryTemplateSize, float* globalScores) {

	// Set up Look-up tables
	LUT_C templateTags;
	LUT_NP numberOfValuesToPick;

	/*if (templateTags.canBeRead(INPUT_FILE_TEMPLATE_TAGS))
		templateTags.read(INPUT_FILE_TEMPLATE_TAGS);
	else*/
		templateTags.fill(templateSizes, templatesNumber);

	/*if (numberOfValuesToPick.canBeRead(INPUT_FILE_NP_VALS))
		numberOfValuesToPick.read(INPUT_FILE_NP_VALS);
	else*/
		numberOfValuesToPick.fill();

	CUDAArray<int> bucketMatrix = constructBuckets(similaritiesDatabase, templatesNumber, templateTags, queryTemplateSize);
	
	computeGlobalScores(bucketMatrix, templateSizes, queryTemplateSize, numberOfValuesToPick, globalScores);

	templateTags.free();
	numberOfValuesToPick.free();
}

void getGlobalScoresFloat(float* similaritiesDatabase, int templatesNumber, short* templateSizes, short queryTemplateSize, float* globalScores) {

	// Set up Look-up tables
	LUT_C templateTags;
	LUT_NP numberOfValuesToPick;

	/*if (templateTags.canBeRead(INPUT_FILE_TEMPLATE_TAGS))
		templateTags.read(INPUT_FILE_TEMPLATE_TAGS);
	else*/
		templateTags.fill(templateSizes, templatesNumber);

	/*if (numberOfValuesToPick.canBeRead(INPUT_FILE_TEMPLATE_TAGS))
		numberOfValuesToPick.read(INPUT_FILE_NP_VALS);
	else*/
		numberOfValuesToPick.fill();

	CUDAArray<int> bucketMatrix = constructBuckets(similaritiesDatabase, templatesNumber, templateTags, queryTemplateSize);

	computeGlobalScores(bucketMatrix, templateSizes, queryTemplateSize, numberOfValuesToPick, globalScores);

	templateTags.free();
	numberOfValuesToPick.free();
}

template<class T>
CUDAArray<int> constructBuckets(T* similaritiesDatabase, int templatesNumber, const LUT_C &templateTags, short databaseHeight) {
	cudaError_t cudaStatus;

	CUDAArray<int> bucketMatrix(databaseHeight, templatesNumber);
	cudaStatus = cudaMemset2D(bucketMatrix.cudaPtr, bucketMatrix.Stride, 0, bucketMatrix.Width * sizeof(short), bucketMatrix.Height);

	// Copy database to GPU

	CUDAArray<T> gpuDB(similaritiesDatabase, templateTags.getSize(), databaseHeight);

	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(templateTags.getSize(), blockSize.x), ceilMod(databaseHeight, blockSize.y));
	quantify<<<gridSize, blockSize>>>(gpuDB, bucketMatrix, templateTags);

	cudaStatus = cudaGetLastError();
	cudaStatus = cudaDeviceSynchronize();

	gpuDB.Dispose();

	return bucketMatrix;
}

void computeGlobalScores(CUDAArray<int> bucketMatrix, short* templateSizes, short queryTemplateSize, const LUT_NP &numberOfValuesToPick, float cpuGlobalScores[]) {
	cudaError_t cudaStatus;

	// Texture perhaps?
	short* gpuTemplateSizes = 0;
	cudaStatus = cudaMalloc(&gpuTemplateSizes, bucketMatrix.Height * sizeof(short));
	cudaStatus = cudaMemcpy(gpuTemplateSizes, templateSizes, bucketMatrix.Height * sizeof(short), cudaMemcpyHostToDevice);

	float* globalScores = 0;
	cudaStatus = cudaMalloc(&globalScores, bucketMatrix.Height * sizeof(float));

	dim3 blockSize = bucketMatrix.Width;
	dim3 gridSize = ceilMod(bucketMatrix.Height, blockSize.x);

	size_t subBucketsPitch = getPitch<short>(bucketMatrix.Width);

	cudaComputeGlobalScores<<<gridSize, blockSize, blockSize.x * subBucketsPitch>>>
		(globalScores, bucketMatrix, subBucketsPitch / sizeof(short), queryTemplateSize, gpuTemplateSizes, numberOfValuesToPick);
	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(cpuGlobalScores, globalScores, bucketMatrix.Height * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(globalScores);
	cudaFree(gpuTemplateSizes);
}


// CPU Implementation

template<class T>
T* linearizeDoublePointer(T** matrix, size_t wOffset, size_t height, size_t width) {
	T* lin = new T[height*width];
	size_t bytesInRow = width * sizeof(T);
	for (size_t i = 0; i < height; ++i)
		memcpy(lin + i * width, matrix[i] + wOffset, bytesInRow);
	return lin;
}

// Example: {4,2,3}-->{4,4,4,4,2,2,3,3,3}
void spreadNumbers(short* src, int* dst, size_t srcLength) {
	size_t k = 0;
	for (size_t i = 0; i < srcLength; ++i)
	for (short j = 0; j < src[i]; ++j)
		dst[k++] = i;
}

template<class T>
size_t getPitch(size_t width) {
	T* testPtr = 0;
	size_t pitch;
	cudaMallocPitch(&testPtr, &pitch, width, 1);
	cudaFree(testPtr);
	return pitch;
}


// GPU Implementation

__global__ void quantify(CUDAArray<float> src, CUDAArray<int> dst, LUT_C templateTags) {
	size_t x = defaultRow(),
		y = defaultColumn();
	if ((x < src.Height) && (y < src.Width)) {
		int level = (int)(quantLevels * src.At(x, y));
		atomicAdd(dst.cudaPtr + (dst.Stride / sizeof(int)) * templateTags[y] + level, 1);
	}
}
__global__ void quantify(CUDAArray<short> src, CUDAArray<int> dst, LUT_C templateTags) {
	size_t x = defaultRow(),
		y = defaultColumn();
	if ((x < src.Height) && (y < src.Width)) {
		int level = (int)src.At(x, y);
		atomicAdd(dst.cudaPtr + (dst.Stride / sizeof(int)) * templateTags[y] + level, 1);
	}
}

__global__ void cudaComputeGlobalScores(float* globalScores, CUDAArray<int> bucketMatrix, size_t subBMWidth, short queryTemplateSize, short* templateSizes, LUT_NP numberOfValuesToPick) {

	// Subcopy bucket matrix to shared memory

	size_t hOffset = blockDim.x * blockIdx.x;
	extern __shared__ short subBucketMatrix[];
	size_t rowsNumber = min(blockDim.x, bucketMatrix.Height - hOffset);

	for (size_t i = 0; i < rowsNumber; ++i)
		subBucketMatrix[subBMWidth * i + threadIdx.x] = (short)bucketMatrix.At(hOffset + i, threadIdx.x);
	__syncthreads();


	// Count sum of the lowest values

	if (threadIdx.x < rowsNumber) {
		size_t currentTemplate = defaultColumn();
		const short np = numberOfValuesToPick[min(templateSizes[currentTemplate], queryTemplateSize)];
		short currentLevel = 0,
			canYetTake = np,
			sumOfLowest = 0;
		while ((currentLevel < bucketMatrix.Width) && (canYetTake > 0)) {
			short takeNow = min(subBucketMatrix[threadIdx.x * subBMWidth + currentLevel], canYetTake);
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

	for (int i = 0; i < height; ++i)
	for (int j = 0; j < queryTemplateSize; ++j)
		fin_database >> linDB[i * queryTemplateSize + j];

	auto error = cudaSetDevice(0);

	// auto linDB = linearizeDoublePointer(DB, 0, queryTemplateSize, width);
	
	float *scores = new float[templatesNumber];
	getGlobalScoresFloat(linDB, templatesNumber, templateSizes, queryTemplateSize, scores);
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
#define _USE_MATH_DEFINES 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "CUDAArray.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "ImageLoading.cuh"
#include "OrientationField.cuh"
#include "math_constants.h"
#include "ImageEnhancement.cuh"
#include "Filter.cuh"
#include "Convolution.cuh"
const int l = 32;

extern "C"
{
	__declspec(dllexport) void GetFrequency(float* res, float* image, int height, int width, float* orientMatrix, int interpolationFilterSize = 7,
		int interpolationSigma = 1, int lowPassFilterSize = 19, int lowPassFilterSigma = 3, int w = 16);
}

__device__ void CalculateSignature(float** res, CUDAArray<float>* img, float angle, int x, int y, int w)
{
	float signature[l];
	float angleSin = sin(angle);
	float angleCos = cos(angle);
	for (int k = 0; k < l; k++)
	{
		signature[k] = 0;
		for (int d = 0; d < w; d++)
		{
			int indX = (int)(x + (d - w / 2) * angleCos + (k - l / 2) * angleSin);
			int indY = (int)(y + (d - w / 2) * angleSin + (l / 2 - k) * angleCos);
			if ((indX < 0) || (indY < 0) || (indX >= img->Width) || (indY >= img->Height))
				continue;
			signature[k] += img->At(indX, indY);
		}
		signature[k] /= w;
	}
	*res = signature;
}

__global__ void CalculateFrequencyPixel(CUDAArray<float> res, CUDAArray<float> img, CUDAArray<float> orientMatr, int w)
{
	int column = defaultColumn();
	int row = defaultRow();
	if ((column < img.Width) && (row < img.Height)) {
		
		float* signature;
		CalculateSignature(&signature, &img, orientMatr.At(column, row), column, row, w);

		int prevMin = -1;
		int lengthsSum = 0;
		int summandNum = 0;

		for (int i = 1; i < l - 1; i++)
		{
			//In comparison below there has to be non-zero value so that we would be able to ignore minor irrelevant pits of black.
			if ((signature[i - 1] - signature[i] > 0.5) && (signature[i + 1] - signature[i] > 0.5))
			{
				if (prevMin != -1)
				{
					lengthsSum += i - prevMin;
					summandNum++;
					prevMin = i;
				}
				else
				{
					prevMin = i;
				}
			}
		}
		float frequency = (float)summandNum / lengthsSum;
		if ((lengthsSum <= 0) || (frequency > 1.0 / 3.0) || (frequency < 0.04))
			frequency = -1;
		res.SetAt(row, column, frequency);
		
	}
}

void CalculateFrequency(float* res, float* image, int height, int width, float* orientMatrix, int w)
{
	CUDAArray<float> img               = CUDAArray<float>(image, width, height);
	CUDAArray<float> frequencyMatrix   = CUDAArray<float>(width, height);
	CUDAArray<float> orientationMatrix = CUDAArray<float>(orientMatrix, width, height);

	dim3 blockSize = dim3 (defaultThreadCount, defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(width, defaultThreadCount), ceilMod(height, defaultThreadCount));

	CalculateFrequencyPixel<<<gridSize, blockSize>>>(frequencyMatrix, img, orientationMatrix, w);
	frequencyMatrix.GetData(res);
}

__global__ void InterpolatePixel(CUDAArray<float> frequencyMatrix, CUDAArray<float> result, bool* needMoreInterpolationFlag, CUDAArray<float> filter, int w)
{
	int row    = defaultRow();
	int column = defaultColumn();

	int height = frequencyMatrix.Height;
	int width  = frequencyMatrix.Width;

	*needMoreInterpolationFlag = false;

	if (row < height && column < width) {
		if (frequencyMatrix.At(row, column) == -1.0)
		{
			int center = filter.Width / 2;
			int upperCenter = (filter.Width & 1) == 0 ? center - 1 : center;

			float numerator = 0;
			float denominator = 0;
			for (int drow = -upperCenter; drow <= center; drow++)
			{
				for (int dcolumn = -upperCenter; dcolumn <= center; dcolumn++)
				{
					float filterValue = filter.At(center - drow, center - dcolumn);
					int indexRow = row + drow * w;
					int indexColumn = column + dcolumn * w;

					if (indexRow < 0)    indexRow = 0;
					if (indexColumn < 0) indexColumn = 0;
					if (indexRow >= height)   indexRow = height - 1;
					if (indexColumn >= width) indexColumn = width - 1;

					float freqVal = frequencyMatrix.At(indexRow, indexColumn);
					//Mu:
					float freqNumerator = freqVal;
					if (freqNumerator <= 0) freqNumerator = 0;
					//Delta:
					float freqDenominator = freqVal;
					if (freqDenominator + 1 <= 0) freqDenominator = 0;
					else freqDenominator = 1;

					numerator += filterValue * freqNumerator;
					denominator += filterValue * freqDenominator;
				}
			}
			float freqBuf = numerator / denominator;
			if (freqBuf != freqBuf || freqBuf > 1.0 / 3.0 || freqBuf < 0.04)
			{
				freqBuf = -1;
				*needMoreInterpolationFlag = true;
			}
			result.SetAt(row, column, freqBuf);
		}
		else
			result.SetAt(row, column, frequencyMatrix.At(row, column));
	}
}

void Interpolate(int imgWidth, int imgHeight, float* res, bool* needMoreInterpolationFlag, float* frequencyMatr, int filterSize, float sigma, int w)
{
	CUDAArray<float> result = CUDAArray<float>(imgWidth, imgHeight);
	CUDAArray<float> frequencyMatrix = CUDAArray<float>(frequencyMatr, imgWidth, imgHeight);

	CUDAArray<float> filter = CUDAArray<float>(MakeGaussianFilter(filterSize, sigma), filterSize, filterSize);

	bool* dev_needMoreInterpolationFlag;
	cudaMalloc((void**)&dev_needMoreInterpolationFlag, sizeof(bool));

	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
	dim3 gridSize = dim3(ceilMod(imgWidth, defaultThreadCount), ceilMod(imgHeight, defaultThreadCount));
	InterpolatePixel << <gridSize, blockSize >> >(frequencyMatrix, result, dev_needMoreInterpolationFlag,filter, w);

	cudaMemcpy(needMoreInterpolationFlag, dev_needMoreInterpolationFlag, sizeof(bool), cudaMemcpyDeviceToHost);
	if (needMoreInterpolationFlag)
		Interpolate(imgWidth, imgHeight, res, needMoreInterpolationFlag, result.GetData(), filterSize, sigma, w);

	result.GetData(res);
}

void FilterFrequencies(int imgWidth, int imgHeight, float* res, float* frequencyMatr, int filterSize, float sigma, int w)
{
	CUDAArray<float> result = CUDAArray<float>(imgWidth, imgHeight);
	CUDAArray<float> frequencyMatrix = CUDAArray<float>(frequencyMatr, imgWidth, imgHeight);

	CUDAArray<float> lowPassFilter = CUDAArray<float>(MakeGaussianFilter(filterSize, sigma), filterSize, filterSize);

	Convolve(result, frequencyMatrix, lowPassFilter, w);
	result.GetData(res);
}

void GetFrequency(float* res, float* image, int height, int width, float* orientMatrix, int interpolationFilterSize, 
	int interpolationSigma, int lowPassFilterSize, int lowPassFilterSigma, int w)
{
	float* initialFreq = (float*) malloc(height * width * sizeof(float));
	CalculateFrequency(initialFreq, image, height, width, orientMatrix, w);

	float* interpolatedFreq = (float*)malloc(height * width * sizeof(float));
	Interpolate(width, height, interpolatedFreq, false, initialFreq, interpolationFilterSize, interpolationSigma, w);

	FilterFrequencies(width, height, res, interpolatedFreq, lowPassFilterSize, lowPassFilterSigma, w);
}

void main()
{
	freopen("..\\out.txt", "w+", stdout);
	int width;
	int height;
	int w = 16;
	char* filename = "1_4.bmp";  //Write your way to bmp file
	int* img = loadBmp(filename, &width, &height);
	float* source = (float*)malloc(height*width*sizeof(float));
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			source[i * width + j] = (float)img[i * width + j];
		}
	float* b = (float*)malloc(height * width * sizeof(float));
	float* c = (float*)malloc(height * width * sizeof(float));
	float* d = (float*)malloc(height * width * sizeof(float));
	float* g = (float*)malloc(height * width * sizeof(float));
	float* h = (float*)malloc(height * width * sizeof(float));
	float* orMatr =	OrientationFieldInPixels(source, width, height);

	GetFrequency(g, source, height, width, orMatr);

	CalculateFrequency(b, source, height, width, orMatr, w);
	Interpolate(width, height, d, false, b, 7, 1, 16);
	
	FilterFrequencies(width, height, b, d, 19, 3, 16);
	Enhance(source, width, height, c, orMatr, b, 32, 8);
	Enhance(source, width, height, h, orMatr, g, 32, 8);
	saveBmp("..\\res.bmp", c, width, height);
	saveBmp("..\\res2.bmp", h, width, height);
	free(source);
	free(img);
	free(b);
}
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
		int interpolationSigma = 1, int lowPassFilterSize = 25, int lowPassFilterSigma = 4, int w = 16);
}

__device__ void CalculateSignatureLine(float* res, CUDAArray<float>* img, int width, int height, int index, float angle, int x, int y, int w)
{
	float angleSin = sin(angle);
	float angleCos = cos(angle);

	//int iStart = y - (3 * w / 2 -1);
	//int jStart = x - (3 * w / 2 -1);

	float signature = 0;
	for (int d = 0; d < w; d++)
	{
		int indX = (int)(x + (d - w / 2) * angleCos + (index - l / 2) * angleSin);
		int indY = (int)(y + (d - w / 2) * angleSin + (l / 2 - index) * angleCos);
		if ((indX < 0) || (indY < 0) || (indX >= width) || (indY >= height))
			continue;
	/*	if ((indY - iStart < 0) || (indY - iStart >= 48) || (indX - jStart < 0) || (indX - jStart >= 48))
		{
			signature++;
		}*/
		
		signature += img->At(indX, indY);
			////signature += buffer[1];
			//signature += 1;
	}
	signature /= w;

	*res = signature;
}

__global__ void CalculateFrequencyPixel(CUDAArray<float> res, CUDAArray<float> img, CUDAArray<float> orientMatr, int w)
{
	int column = defaultColumn();
	int row = defaultRow();
	if ((column < img.Width) && (row < img.Height)) {

		int prevMin = -1;
		int lengthsSum = 0;
		int summandNum = 0;

		/*__shared__ float buffer[48][48];
		__shared__ int iStart, iFinish, jStart, jFinish;*/
		//center of the block of even size shall be the left upper pixel of the four central ones.
	/*	int check1 = (row - (w / 2 - 1)) % w;
		int check2 = (column - (w / 2 - 1)) % w;
		if (row == 7 && column == 7)
		{
			summandNum++;
		}*/
		//if (((row - (w / 2 - 1)) % w == 0) && ((column - (w / 2 - 1)) % w == 0))
		//{
		//	iStart = row - ((3 * w) / 2 - 1);
		//	iFinish = row + (3 * w) / 2 + 1;
		//	if (iStart < 0) iStart = 0;
		//	if (iFinish > img.Height) iFinish = img.Height;

		//	jStart = column - ((3 * w) / 2 - 1);
		//	jFinish = column + (3 * w) / 2 + 1;
		//	//if (iStart == -1)
		//	//{
		//	//	jStart--;
		//	//}
		//	if (jStart < 0) jStart = 0;
		//	if (jFinish > img.Width) jFinish = img.Width;
		//	for (int i = iStart; i < iFinish; i++)
		//	for (int j = jStart; j < jFinish; j++)
		//	{
		//		//if (((i - iStart) * (3 * w / 2) + (j - jStart) <= 9 * w * w - 1) && ((i - iStart) * (3 * w / 2) + (j - jStart) >= 0))
		//		//{
		//			buffer[(i - iStart)][(j - jStart)] = img.At(i, j);
		//		//}
		//		//buffer[(i - iStart) * (3 * w / 2) + (j - jStart)] = img.At(i, j);
		//		//lengthsSum = img.At(i, j);
		//	}
		//}
		//__syncthreads();
		float a, b, c;
		CalculateSignatureLine(&a, &img, img.Width, img.Height, 0, orientMatr.At(row, column), column, row, w);
		CalculateSignatureLine(&b, &img, img.Width, img.Height, 1, orientMatr.At(row, column), column, row, w);
		for (int i = 1; i < l - 1; i++)
		{
			CalculateSignatureLine(&c, &img, img.Width, img.Height, i + 1, orientMatr.At(row, column), column, row, w);
			//In comparison below there has to be non-zero value so that we would be able to ignore minor irrelevant pits of black.
			if ((a - b > 0.5) && (c - b  > 0.5))
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
				//	lengthsSum--;
				}
			}
			a = b;
			b = c;
		}
		float frequency = (float)summandNum / lengthsSum;
		if ((lengthsSum <= 0) || (frequency > 1.0f / 3.0f) || (frequency < 0.04f))
			frequency = -1;
		res.SetAt(row, column, frequency);
	}
}

void CalculateFrequency(float* res, float* image, int height, int width, float* orientMatrix, int w)
{
	CUDAArray<float> img               = CUDAArray<float>(image, width, height);
	CUDAArray<float> frequencyMatrix   = CUDAArray<float>(width, height);
	CUDAArray<float> orientationMatrix = CUDAArray<float>(orientMatrix, width, height);

	/*dim3 blockSize = dim3 (defaultThreadCount/2, defaultThreadCount/2);
	dim3 gridSize = dim3(ceilMod(width, (defaultThreadCount/2)), ceilMod(height, (defaultThreadCount/2)));*/

	dim3 blockSize = dim3(16, 16);
	dim3 gridSize = dim3(ceilMod(width, 16), ceilMod(height, 16));
	CalculateFrequencyPixel<<<gridSize, blockSize>>>(frequencyMatrix, img, orientationMatrix, w);
	frequencyMatrix.GetData(res);

	img.Dispose();
	frequencyMatrix.Dispose();
	orientationMatrix.Dispose();
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

	cudaFree(dev_needMoreInterpolationFlag);
	result.Dispose();
	frequencyMatrix.Dispose();
	filter.Dispose();
}

void FilterFrequencies(int imgWidth, int imgHeight, float* res, float* frequencyMatr, int filterSize, float sigma, int w)
{
	CUDAArray<float> result = CUDAArray<float>(imgWidth, imgHeight);
	CUDAArray<float> frequencyMatrix = CUDAArray<float>(frequencyMatr, imgWidth, imgHeight);

	CUDAArray<float> lowPassFilter = CUDAArray<float>(MakeGaussianFilter(filterSize, sigma), filterSize, filterSize);

	Convolve(result, frequencyMatrix, lowPassFilter, w);
	result.GetData(res);

	result.Dispose();
	frequencyMatrix.Dispose();
	lowPassFilter.Dispose();
}

void GetFrequency(float* res, float* image, int height, int width, float* orientMatrix, int interpolationFilterSize, 
	int interpolationSigma, int lowPassFilterSize, int lowPassFilterSigma, int w)
{
	float* initialFreq = (float*) malloc(height * width * sizeof(float));
	CalculateFrequency(initialFreq, image, height, width, orientMatrix, w);

	float* interpolatedFreq = (float*)malloc(height * width * sizeof(float));
	Interpolate(width, height, interpolatedFreq, false, initialFreq, interpolationFilterSize, interpolationSigma, w);

	FilterFrequencies(width, height, res, interpolatedFreq, lowPassFilterSize, lowPassFilterSigma, w);

	free(initialFreq);
	free(interpolatedFreq);
}

void main()
{
	cudaSetDevice(0);
	int width;
	int height;
	int w = 16;
	char* filename = "..//4_8.bmp";  //Write your way to bmp file
	int* img = loadBmp(filename, &width, &height);
	float* source = (float*)malloc(height*width*sizeof(float));
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			source[i * width + j] = (float)img[i * width + j];
		}
	//float* b = (float*)malloc(height * width * sizeof(float));
	//float* c = (float*)malloc(height * width * sizeof(float));
	//float* d = (float*)malloc(height * width * sizeof(float));
	float* g = (float*)malloc(height * width * sizeof(float));
	float* h = (float*)malloc(height * width * sizeof(float));
	float* orMatr =	OrientationFieldInPixels(source, width, height);

	GetFrequency(g, source, height, width, orMatr, 7, 1, 25, 4, 16);
	
	//CalculateFrequency(b, source, height, width, orMatr, w);
	//Interpolate(width, height, d, false, b, 7, 1, 16);
	//
	//FilterFrequencies(width, height, b, d, 19, 3, 16);
	//Enhance(source, width, height, c, orMatr, b, 32, 8);
	Enhance(source, width, height, h, orMatr, g, 32, 8);
//	saveBmp("..\\res.bmp", c, width, height);
	saveBmp("..\\res.bmp", h, width, height);
	free(source);
	free(img);
	free(g);
	free(h);
	cudaDeviceReset();
}
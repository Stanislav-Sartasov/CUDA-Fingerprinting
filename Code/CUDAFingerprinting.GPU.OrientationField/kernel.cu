#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_functions.h>
#include <math_constants.h>
#include <stdio.h>
#include <stdlib.h>
#include "Convolution.cuh"
#include "CUDAArray.cuh"
#include "ImageLoading.cuh"
//extern "C"{
//	__declspec(dllexport) void makeOrientationField(float* img, int imgWidth, int imgHeight, float* orField, int regionSize, int overlap);
//}
//
//// smoothes the orientation field components and returns it as an array of angles
//__global__ void	makeSmoothedOrientationField(
//		CUDAArray<float> orField, 
//		CUDAArray<float> xPointer, 
//		CUDAArray<float> yPointer)
//{
//	int row = defaultRow();
//	int column = defaultColumn();
//	if(row < orField.Height && column < orField.Width)
//	{
//		float xMean = 0.0f, yMean = 0.0f;
//		float xCount = 0.0f, yCount = 0.0f;
//
//		//for(int i = -1; i < 2; i++)
//		//{
//		//	int x1 = row + i;
//		//	if(x1 < 0 || x1 >= orField.Height) continue;
//		//	for(int j= -1; j < 2; j++)
//		//	{
//		//		int y1 = column + j;
//		//		if(y1 < 0 || y1 >= orField.Width) continue;
//		//	
//		//		xCount++;
//		//		yCount++;
//		//		xMean += yPointer.At(x1, y1);
//		//		yMean += xPointer.At(x1, y1);
//		//	}
//		//}
//
//		//float xComp = xMean / xCount;
//		//float yComp = yMean / yCount;
//
//		float xComp = xPointer.At(row, column);
//		float yComp = yPointer.At(row, column);
//		
//		float result = 0.0f;
//		if (__isnanf(xComp) || __isnanf(yComp)) result = 0.0f;
//
//		else if (!(xComp == 0.0f && yComp == 0.0f))
//		{
//			result = atan2(xComp, yComp);
//			result = result / 2.0f + CUDART_PI_F / 2.0f;
//			if (result> CUDART_PI_F) result -= CUDART_PI_F;
//		}
//		orField.SetAt(row, column, result);
//	}
//}
//
//__global__ void formRawOrientationFieldComponents(
//	CUDAArray<float> sobelX,
//	CUDAArray<float> sobelY,
//	CUDAArray<float> orFieldX,
//	CUDAArray<float> orFieldY,
//	int regionSize, int overlap)
//{
//	int orFieldXDim = sobelX.Width / (regionSize - overlap);
//    int orFieldYDim = sobelX.Height / (regionSize - overlap);
//	int regionColumn = defaultColumn();
//	int regionRow = defaultRow();
//
//	if(regionColumn<orFieldXDim&&regionRow<orFieldYDim)
//	{
//		float G = 0.0f,
//		 	  Gxy=0.0f;
//		for (int u = 0; u < regionSize; u++)
//        {
//            for (int v = 0; v < regionSize; v++)
//            {
//                 int mColumn = regionColumn*(regionSize - overlap) + u;
//                 int mRow = regionRow*(regionSize - overlap) + v;
//                 if (mColumn > sobelX.Width || mRow > sobelX.Height) continue;
//				 
//				 float sx = sobelX.At(mRow, mColumn);
//				 float sy = sobelY.At(mRow, mColumn);
//                 Gxy += 2.0f*sx*sy;
//                 G += sx*sx -sy*sy;
//            }
//        }
//
//		orFieldX.SetAt(regionRow, regionColumn, Gxy);
//        orFieldY.SetAt(regionRow, regionColumn, G);
//	}
//}
//
//void SaveArray(float* arTest, int width, int height, const char* fname)
//{
//	FILE* f = fopen(fname,"wb");
//	fwrite(&width,sizeof(int),1,f);
//	fwrite(&height,sizeof(int),1,f);
//	for(int i=0;i<width*height;i++)
//	{
//		float value = (float)arTest[i];
//		int result = fwrite(&value,sizeof(float),1,f);
//		result++;
//	}
//	fclose(f);
//	free(arTest);
//}
//
//// exported function that creates the orientation field and searches for the core point
//void makeOrientationField(float* img, int imgWidth, int imgHeight, float* orField, int regionSize, int overlap)
//{
//	cudaError_t cudaStatus = cudaSetDevice(0);
//	
//	CUDAArray<float> source = CUDAArray<float>(img, imgWidth, imgHeight);
//
//	// Sobel :	  
//	CUDAArray<float> xGradient = CUDAArray<float>(imgWidth,imgHeight);
//	CUDAArray<float> yGradient = CUDAArray<float>(imgWidth,imgHeight);
//
//	cudaStatus = cudaGetLastError();
//	 
//	float xKernelCPU[3][3] = {{1,0,-1},
//							{2,0,-2},
//							{1,0,-1}};
//
//	CUDAArray<float> xKernel = CUDAArray<float>((float*)&xKernelCPU,3,3);	  
//
//	float yKernelCPU[3][3] = {{1,2,1},
//							{0,0,0},
//							{-1,-2,-1}};
//
//	CUDAArray<float> yKernel = CUDAArray<float>((float*)&yKernelCPU,3,3);	  
//
//	Convolve(xGradient, source, xKernel);
//	Convolve(yGradient, source, yKernel);
//
//	xKernel.Dispose();
//	yKernel.Dispose();
//
//    int orFieldWidth = imgWidth / (regionSize - overlap);
//    int orFieldHeight = imgHeight / (regionSize - overlap);
//
//	CUDAArray<float> orFieldCuda = CUDAArray<float>(orFieldWidth, orFieldHeight);
//	CUDAArray<float> orFieldX = CUDAArray<float>(orFieldWidth, orFieldHeight);
//	CUDAArray<float> orFieldY = CUDAArray<float>(orFieldWidth, orFieldHeight);
//
//	dim3 blockSize = dim3(defaultThreadCount, defaultThreadCount);
//	dim3 gridSize = dim3(ceilMod(orFieldWidth, defaultThreadCount), ceilMod(orFieldHeight, defaultThreadCount));
//
//	formRawOrientationFieldComponents<<<gridSize, blockSize>>>(
//		xGradient, yGradient, orFieldX, orFieldY, regionSize, overlap);
//
//	cudaError_t error = cudaGetLastError();
//
//	makeSmoothedOrientationField<<<gridSize, blockSize>>>(
//		orFieldCuda, orFieldX, orFieldY);
//
//	error = cudaGetLastError();
//	
//	SaveArray(orFieldCuda.GetData(), orFieldCuda.Width, orFieldCuda.Height, "C:\\temp\\orField.bin");
//
//	xGradient.Dispose();
//	yGradient.Dispose();
//
//	orFieldCuda.GetData(orField);
//	orFieldCuda.Dispose();
//
//	orFieldX.Dispose();
//	orFieldY.Dispose();
//}
//
//CUDAArray<float> loadImage(const char* name, bool sourceIsFloat = false)
//{
//	FILE* f = fopen(name,"rb");
//			
//	int width;
//	int height;
//	
//	fread(&width,sizeof(int),1,f);			
//	fread(&height,sizeof(int),1,f);
//	
//	float* ar2 = (float*)malloc(sizeof(float)*width*height);
//
//	if(!sourceIsFloat)
//	{
//		int* ar = (int*)malloc(sizeof(int)*width*height);
//		fread(ar,sizeof(int),width*height,f);
//		for(int i=0;i<width*height;i++)
//		{
//			ar2[i]=ar[i];
//		}
//		
//		free(ar);
//	}
//	else
//	{
//		fread(ar2,sizeof(float),width*height,f);
//	}
//	
//	fclose(f);
//
//	CUDAArray<float> sourceImage = CUDAArray<float>(ar2,width,height);
//
//	free(ar2);		
//
//	return sourceImage;
//}

int main()
{
	//BMPHeader header;
	//int* img = loadBmp(&header, "C:\\temp\\DB2_bmp\\1_1.bmp");
	//saveBmp(img, &header, "C:\\temp\\SaveTestfire.bmp");
	//free(img);
	return 0;
}
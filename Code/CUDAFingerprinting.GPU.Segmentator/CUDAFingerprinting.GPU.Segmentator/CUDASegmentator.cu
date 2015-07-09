#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
#include "Convolution.cuh"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define edge 100
#define pixEdge 120

/*void SobelFilter(float* pic, int width, int height, CUDAArray<float> &matrix)
{
	Gx[0][0] = -1; Gx[0][1] = 0; Gx[0][2] = 1;
	Gx[1][0] = -2; Gx[1][1] = 0; Gx[1][2] = 2;
	Gx[2][0] = -1; Gx[2][1] = 0; Gx[2][2] = 1;

	int **Gy;
	Gx[0][0] = 1; Gx[0][1] = 2; Gx[0][2] = 1;
	Gx[2][0] = -1; Gx[2][1] = -2; Gx[2][2] = -1;

	int x = threadIdx.x;
	int y = threadIdx.y;

    while ( (x < width - 1) && (y < height - 1) )
    {
        double sumX = pic[(x - 1) * width + (y - 1)] * Gx[0][0] + pic[(x + 1) * width + (y - 1)] * Gx[0][2] +
                        pic[(x - 1) * width + y] * Gx[1][0] + pic[(x + 1) * width + y] * Gx[1][2] +
                        pic[(x - 1) * width + (y + 1)] * Gx[2][0] + pic[(x + 1) * width + (y + 1)] * Gx[2][2];
        double sumY = pic[(x - 1) * width + (y - 1)] * Gy[0][0] + pic[x * width + (y - 1)] * Gy[0][1] + pic[(x + 1) * width + (y - 1)] * Gy[0][2] +
            pic[(x - 1) * width + (y + 1)] * Gy[2][0] + pic[x * width + (y + 1)] * Gy[2][1] + pic[(x + 1) * width + (y + 1)] * Gy[2][2];
        double sqrtXY = sqrt(sumX * sumX + sumY * sumY);

		matrix.SetAt (x, y, sqrtXY);
		x += blockDim.x * gridDim.x;
		y += blockDim.y * gridDim.y;
	}
};*/

__global__ void cudaSegmentate(CUDAArray<float> segmentation, int defaultBlockSize)
{
	int column = defaultColumn();
	int row = defaultRow();

	int tX = threadIdx.x;
	int tY = threadIdx.y;

	__shared__ int sumX = 0;
	__shared__ int sumY = 0;

	while ( tY < defaultBlockSize )
	{
		sumX = 0;
		while ( tX < defaultBlockSize )
		{
			sumX += segmentation.At(tX, tY);
		}
		sumY += sumX;
		__shared__ float average = sumY / ( defaultBlockSize * defaultBlockSize );

		if ( average >= edge )
		{
			//TODO: Assignment to result matrix 0 or 1
		}
	}


    /*while ( (x <= width - 16) && (y <= height - 16) )
    {
        double averageColor = 0;

        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {

                averageColor += matrix.At(x + i, y + j);
            }
        }

        averageColor /= 256;

        if (averageColor >= edge)
        {
            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    if (matrix.At(x + i, y + j) >= pixEdge)
                    {
						matrix.SetAt (x + i, y + j, 1);
                    }
                    else
                    {
                        matrix.SetAt (x + i, y + j, 0);
                    }
                }
            }
		}

		x += blockDim.x * gridDim.x;
		y += blockDim.y * gridDim.y;
    }

    //Processing the bottom of the image*/

	/*x = threadIdx.x;

    if (height % 16 != 0)
    {
        while (x <= width - 16)
        {
            double averageColor = 0;

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < height % 16; j++)
                {
                    //Color pixColor = block.GetPixel(i, j);
                    averageColor += matrix.At(x + i, height - (height % 16) + j);
                }
            }

            averageColor /= (16 * (height % 16));

            if (averageColor >= edge)
            {
                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < height % 16; j++)
                    {
                        if (matrix.At(x + i, height - (height % 16) + j) >= pixEdge)
                        {
							matrix.SetAt (x + i, height - ( height % 16 ) + j, 1);
                        }
                        else
                        {
                            matrix.SetAt (x + i, height - ( height % 16 ) + j, 0);
                        }
                    }
                }
            }
        }

		x += blockDim.x * gridDim.x;
    }

    //Processing the right side of the image

	y = threadIdx.y + blockIdx.y * blockDim.y;

    if (width % 16 != 0)
    {
        while ( y <= height - 16 )
        {
            double averageColor = 0;

            for (int i = 0; i < width % 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    averageColor += matrix.At(width - (width % 16) + i, y + j);
                }
            }

            averageColor /= (16 * (width % 16));

            if (averageColor >= edge)
            {
                for (int i = 0; i < width % 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        if (matrix.At(width - (width % 16) + i, y + j) >= pixEdge)
                        {
							matrix.SetAt (width - ( width % 16 ) + i, y + j, 1);
                        }
                        else
                        {
                            matrix.SetAt (width - ( width % 16 ) + i, y + j, 0);
                        }
                    }
                }
            }
        }

		y += blockDim.y * gridDim.y;
    }

    //Processing the right bottom square of the image

	x = threadIdx.x;
	y = threadIdx.y;

    if (width % 16 != 0 && height % 16 != 0)
    {

        double averageColor2 = 0;

        for (int i = 0; i < 16; i++)
        {
            for (int j = 0; j < 16; j++)
            {
                averageColor2 += matrix.At(width - (width % 16) + i, height - (height % 16) + j);
            }
        }

        averageColor2 /= ((width % 16) * (height % 16));

        if (averageColor2 >= edge)
        {

            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    if (matrix.At(width - (width % 16) + i, height - (height % 16) + j) >= pixEdge)
                    {
						matrix.SetAt (width - ( width % 16 ) + i, height - ( height % 16 ) + j, 1);
					}
                    else
                    {
                        matrix.SetAt (width - ( width % 16 ) + i, height - ( height % 16 ) + j, 0);
                    }
                }
            }

        }

    }*/

}

void Segmentate (CUDAArray<float> segmentation, CUDAArray<float> source, int defaultBlockSize)
{
	dim3 blockSize = dim3(defaultBlockSize, defaultBlockSize);
	dim3 gridSize = dim3(ceilMod(source.Width, defaultBlockSize), ceilMod(source.Height, defaultBlockSize));
	cudaSegmentate << <gridSize, blockSize >> >(segmentation, defaultBlockSize);
}

//void BWPicture (int* pic, int width, int height, CUDAArray<float> intMatrix)
//{
//	//Creating Black-White Bitmap on the basis of Matrix
//	
//	int* newPic;
//
//	int x = threadIdx.x;
//	int y = threadIdx.y;
//
//	while (x < width && y < height)
//	{
//		newPic [x * width + y]= (int)intMatrix.At(x, y);
//		x += blockDim.x * gridDim.x;
//		y += blockDim.y * gridDim.y;
//	}
//
//	saveBmp ("newPic.bmp", newPic, width, height);
//}

int main()
{
	const int defaultBlockSize = 16;
	int width, height;
	char* filename = "../1_1.bmp";
	int* pic = loadBmp (filename, &width, &height);

	cudaSetDevice (0);

	float* arr;
	CUDAArray<float> matrix = CUDAArray<float> (arr, width, height); //Saves results of Sobel filter

	//Sobel Filter

	CUDAArray<float> source((float*)pic, width, height);
	CUDAArray<float> Segmentation(source.Width, source.Height);

	float filterXLinear[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
	float filterYLinear[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };

	CUDAArray<float> filterX(filterXLinear, 3, 3);
	CUDAArray<float> filterY(filterYLinear, 3, 3);
	
	CUDAArray<float> Gx(width, height);
	CUDAArray<float> Gy(width, height);
	Convolve(Gx, source, filterX);
	Convolve(Gy, source, filterY);

	Segmentate (Segmentation, source, defaultBlockSize);
	
	//BWPicture (pic, width, height, matrix);

	cudaFree(pic);
	cudaFree (arr);

	return 0;
}
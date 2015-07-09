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

void SobelFilter(int* pic, int width, int height, CUDAArray<float> &matrix)
{
	int capacity = ( width - 1 ) * ( height - 1 );

	int **Gx;
	Gx[0][0] = -1; Gx[0][1] = 0; Gx[0][2] = 1;
	Gx[1][0] = -2; Gx[1][1] = 0; Gx[1][2] = 2;
	Gx[2][0] = -1; Gx[2][1] = 0; Gx[2][2] = 1;

	int **Gy;
	Gx[0][0] = 1; Gx[0][1] = 2; Gx[0][2] = 1;
	Gx[2][0] = -1; Gx[2][1] = -2; Gx[2][2] = -1;

    //Using Sobel Operator

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

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
};

//CUDAArray<int> Segmentate(int width, int height, int** matrix)
__global__ void Segmentate(int width, int height, CUDAArray<float> &matrix)
{
	//CUDAArray<int> Matrix = CUDAArray<int>(width, height);

    //Creating Matrix with '1' for white and '0' for black

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

    while ( (x <= width - 16) && (y <= height - 16) )
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

    //Processing the bottom of the image

	x = threadIdx.x + blockIdx.x * blockDim.x;

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

	x = threadIdx.x + blockIdx.x * blockDim.x;
	y = threadIdx.y + blockIdx.y * blockDim.y;

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
    }

	//return Matrix;
}

void BWPicture (int* pic, int width, int height, CUDAArray<float> intMatrix)
{
	//Creating Black-White Bitmap on the basis of Matrix
	
	int* newPic;

	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	while (x < width && y < height)
	{
		newPic [x * width + y]= (int)intMatrix.At(x, y);
		x += blockDim.x * gridDim.x;
		y += blockDim.y * gridDim.y;
	}

	saveBmp ("newPic.bmp", newPic, width, height);
}

int main()
{
	int width, height;
	char* filename = "../1_1.bmp";
	int* pic = loadBmp (filename, &width, &height);

	float* arr;
	CUDAArray<float> matrix = CUDAArray<float> (arr, width, height); //Saves results of Sobel filter

	SobelFilter (pic, width, height, matrix);
	Segmentate (width, height, matrix);
	
	BWPicture (pic, width, height, matrix);

	cudaFree(pic);
	cudaFree (arr);

	return 0;
}
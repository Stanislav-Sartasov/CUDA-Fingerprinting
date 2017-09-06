#include "kernel.cuh"
#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "constsmacros.h"
#include <stdlib.h>
#include <math.h>
#include "ImageLoading.cu"
//#include "CUDAArray.cuh"
#include <float.h>
#include "OrientationField.cu"
#include "Convolution.cu"

#define M_PI 3.14159265358979323846
#define Pi4 (M_PI / 4);

class RidgeOnLine
{
	
private:
	int _step;
	int _sizeSection;
	int _height;
	int _width;

public:
	RidgeOnLine()
	{
		
	}

	~RidgeOnLine()
	{

	}
};

__device__ __host__
Point NewPoint(int x, int y)
{
	Point newP;
	newP.x = x;
	newP.y = y;
	return newP;
}

__device__ void AddMinutiae(CUDAArray<int>* countOfMinutiae, CUDAArray<Minutiae>* minutiaes, Minutiae minutiae, int* indexOfMinutiae)
{
	//minutiaes.At(blockIdx.x * gridDim.x + blockIdx.y, 0)->Add(minutiae);
	//printf("Adding a minutiae\n");
	minutiaes->SetAt(0, *indexOfMinutiae, minutiae);
	int past = countOfMinutiae->At(0, blockIdx.x * gridDim.x + blockIdx.y);
	countOfMinutiae->SetAt(0, blockIdx.x * gridDim.x + blockIdx.y, past + 1);
	*countOfMinutiae++;
	//printf("%d %d: Was %d. Must be %d, but in real %d\n", blockIdx.x, blockIdx.y, past, past + 1, countOfMinutiae->At(blockIdx.x * gridDim.x + blockIdx.y, 0));
}

__device__ bool OutOfImage(CUDAArray<float> image, int x, int y, int partX, int partY)
{
	//return (x < 0) || (y < 0) || (y >= image.Width) || (x >= image.Height);
	return (x < blockIdx.x * partX) || (y < blockIdx.y * partY) || (x >= (blockIdx.x + 1) * partX) || (y >= (blockIdx.y + 1) * partY) || (y >= image.Width) || (x >= image.Height);
}

__device__ void NewSection(int sx, int sy, Direction direction, CUDAArray<float> image, CUDAArray<float> orientationField, 
	Point* section, float* sectionAngle, int* centerSection, bool* flag, int size, int partX, int partY)
{
	int lengthWings = size / 2;

	for (int i = 0; i < size; i++)
	{
		section[i] = NewPoint(-1, -1);

	}

	int x = sx;
	int y = sy;

	int lEnd = lengthWings;
	int rEnd = lEnd;

	bool rightE = false;
	bool leftE = false;

	float angle = orientationField.At(x, y) + M_PI / 2;

	for (int i = 1; i <= lengthWings; i++)
	{
		int xs = (int)(x - i * cos(angle));
		int ys = (int)(y - i * sin(angle));
		int xe = (int)(x + i * cos(angle));
		int ye = (int)(y + i * sin(angle));

		if (!OutOfImage(image, xs, ys, partX, partY) && (image.At(xs, ys) < 15) && !rightE)
		{
			section[lengthWings - i] = NewPoint(xs, ys);
			rEnd--;
		}
		else
		{
			rightE = true;
		}

		if (!OutOfImage(image, xe, ye, partX, partY) && (image.At(xe, ye) < 15) && !leftE)
		{
			section[lengthWings - i] = NewPoint(xe, ye);
			lEnd--;
		}
		else
		{
			leftE = true;
		}

		*centerSection = (lEnd + rEnd) / 2;

		x = section[*centerSection].x;
		y = section[*centerSection].y;
	}

	angle = orientationField.At(x, y) + direction * M_PI;
	if (angle < 0) angle += 2 * M_PI;

	if (*flag){
		if (abs(*sectionAngle - angle) > 0.2 && abs(*sectionAngle - angle) < 6) angle + M_PI;
	}
	else *flag = true;

	*sectionAngle = angle;
}



__device__ bool CheckAndDeleteFalseMinutia(Minutiae minutia)
{
	return false;
}

__device__ Point MakeStep(CUDAArray<float> image, Point* section, int* centerSection, float* sectionAngle, int step, int partX, int partY)
{
	int x = section[*centerSection].x;
	int y = section[*centerSection].y;

	float dx = (float)x + (float)step * cos(*sectionAngle);
	float dy = (float)y + (float)step * sin(*sectionAngle);

	x = (int)(dx >= 0 ? dx + 0.5 : dx - 0.5);
	y = (int)(dy >= 0 ? dy + 0.5 : dy - 0.5);

	return OutOfImage(image, x, y, partX, partY) ? NewPoint(-1, -1) : NewPoint(x, y);
}

__device__ MinutiaeType CheckStopCriteria(CUDAArray<float> image, CUDAArray<bool> visited, Point* section, int* centerSection, int threshold = 20)
{
	int x = section[*centerSection].x;
	int y = section[*centerSection].y;

	if (visited.At(x, y))
		return Intersection;
	if (image.At(x, y) > threshold)
		return LineEnding;

	return NotMinutiae;
}

__device__ Point queue[30 * 15 * 15];

__device__ void Paint(CUDAArray<float> image, CUDAArray<bool> visited, Point* oldSection, Point* section, int size, int partX, int partY)
{
	//Queue* queue = new Queue;
	int shift = 30 * (blockIdx.x * gridDim.x + blockIdx.y);
	int rPointer = 0, lPointer = 0;
	//queue = (Point*)malloc(32 * 32 * sizeof(Point));
	Point v1, v2;

	int x1 = -1, x2 = -1, y1 = -1, y2 = -1, x_a, y_a;

	for (int i = 0; i < size; i++)
	{
		if (oldSection[i].x == -1) continue;

		if (x1 == -1)
		{
			x1 = oldSection[i].x;
			y1 = oldSection[i].y;
		}

		x2 = oldSection[i].x;
		y2 = oldSection[i].y;

		visited.SetAt(oldSection[i].x, oldSection[i].y, true);
		queue[shift + rPointer] = oldSection[i];
		rPointer++;
	}

	v1 = NewPoint(x2 - x1, y2 - y1);
	x_a = x1;
	y_a = y1;

	x1 = -1;
	y1 = -1;
	x2 = -1;
	y2 = -1;

	for (int i = 0; i < size; i++)
	{
		if (section[i].x == -1) continue;

		if (x1 == -1)
		{
			x1 = section[i].x;
			y1 = section[i].y;
		}

		x2 = section[i].x;
		y2 = section[i].y;

		visited.SetAt(section[i].x, section[i].y, true);
		//printf("%d %d: Point (%d, %d) has been choosed as stop point\n", blockIdx.x, blockIdx.y, i, j);
		queue[shift + rPointer] = section[i];
		rPointer++;
	}

	v2 = NewPoint(x2 - x1, y2 - y1);

	if (v1.x*v2.x + v1.y*v2.y < 0)
	{
		x1 = x2;
		y1 = y2;
		v1 = NewPoint(-v1.x, -v1.y);
	}

	while (rPointer - lPointer > 0)
	{
		Point point = queue[shift + lPointer];
		lPointer++;

		int cX = point.x;
		int cY = point.y;

		for (int i = -1; i < 2; i++)
			for (int j = -1; j < 2; j++)
			{
				if (i == 0 && j == 0) continue;

				int x = cX + i;
				int y = cY + j;

				if (OutOfImage(image, x, y, partX, partY) || visited.At(x, y) || image.At(x, y) > 15) continue;

				Point pointV1 = NewPoint(x_a - x, y_a - y);
				Point pointV2 = NewPoint(x1 - x, y1 - y);

				int skew1 = v1.x*pointV1.y - pointV1.x*v1.y >= 0 ? 1 : -1;
				int skew2 = v2.x*pointV2.y - pointV2.x*v2.y >= 0 ? 1 : -1;

				if (skew1*skew2 < 0)
				{
					queue[shift + rPointer] = NewPoint(x, y);
					rPointer++;
					visited.SetAt(x, y, true);
					//printf("%d %d: Point (%d, %d) has been visited\n", blockIdx.x, blockIdx.y, x, y);
				}
			}
	}
}

__device__ void FollowLine(int x, int y, Direction direction, CUDAArray<float> image, CUDAArray<float> orientationField,
	CUDAArray<bool> visited, CUDAArray<int> countOfMinutiae, CUDAArray<Minutiae> minutiaes,
	Point* section, float* sectionAngle, int* centerSection, bool* flag, int size, int step, int partX, int partY, int* indexOfMinutiae)
{
	NewSection(x, y, direction, image, orientationField, section, sectionAngle, centerSection, flag, size, partX, partY);
	if (section[*centerSection].x == -1) return;

	MinutiaeType type;
	Point point;

	do
	{
		Point* oldSection = new Point[size];
		for (int i = 0; i < size; i++)
			oldSection[i] = section[i];

		point = MakeStep(image, section, centerSection, sectionAngle, step, partX, partY);

		if (point.x == -1) return;

		NewSection(point.x, point.y, direction, image, orientationField, section, sectionAngle, centerSection, flag, size, partX, partY);
		if (section[*centerSection].x == -1) return;

		type = CheckStopCriteria(image, visited, section, centerSection);

		Paint(image, visited, oldSection, section, size, partX, partY);
	} while (type == NotMinutiae);

	Minutiae possMinutiae;
	possMinutiae.x = point.x;
	possMinutiae.y = point.y;
	possMinutiae.angle = *sectionAngle;
	possMinutiae.type = type;

	//printf("Minutia. x = %d y = %d type = %d\n", possMinutiae.x, possMinutiae.y, possMinutiae.type);

	//if (IsDuplicate(possMinutiae)) return;

	if (!CheckAndDeleteFalseMinutia(possMinutiae))
	{
		//printf("Minutia. x = %d y = %d type = %d\n", possMinutiae.x, possMinutiae.y, possMinutiae.type);
		AddMinutiae(&countOfMinutiae, &minutiaes, possMinutiae, indexOfMinutiae);
	}
}

__global__ void FindMinutia(CUDAArray<float> image, CUDAArray<float> orientationField, CUDAArray<bool> visited,
	CUDAArray<int> countOfMinutiae, CUDAArray<Minutiae> minutiaes, 
	const int size, const int step, int colorThreshold = 15)
{
	Point* section = new Point[size];
	float sectionAngle;
	int centerSection;
	bool flag;
	//minutiaes.SetAt(blockIdx.x * gridDim.x + blockIdx.y, 0, new ListOfMinutiae);

	int partX = 32; //image.Height / gridDim.x;
	int partY = 32; //image.Width / gridDim.y;

	int indexOfMinutiae = blockIdx.x * image.Height + blockIdx.y * defaultThreadCount;

	//printf("%d %d %d\n", blockIdx.x, blockIdx.y, threadIdx.x);

	//if (blockIdx.x == 11 && blockIdx.y == 7)
	for (int i = blockIdx.x * partX; i < (blockIdx.x + 1) * partX; i++)
		for (int j = blockIdx.y * partY; j < (blockIdx.y + 1) * partY; j++)
	//for (int i = 0; i < image.Height; i++)
	//	for (int j = 0; j < image.Width; j++)
		{
			if (OutOfImage(image, i, j, partX, partY))
			{
				//printf("Tu-tu. %d %d\n", i, j);
				continue;
			}

			if ((image.At(i, j) >= colorThreshold) || visited.At(i, j)) continue;
			visited.SetAt(i, j, true);
			//if (blockIdx.x == 0 && blockIdx.y == 0) printf("%d %d: Point (%d, %d) has been visited\n", blockIdx.x, blockIdx.y, i, j);

			//printf("%d %d:\n", i, j);

			FollowLine(i, j, Forward, image, orientationField, visited, countOfMinutiae, minutiaes, 
				section, &sectionAngle, &centerSection, &flag, size, step, partX, partY, &indexOfMinutiae);
			FollowLine(i, j, Back, image, orientationField, visited, countOfMinutiae, minutiaes, 
				section, &sectionAngle, &centerSection, &flag, size, step, partX, partY, &indexOfMinutiae);
		}
}

ListOfMinutiae* MergeMinutiaePools(ListOfMinutiae** pools)
{
	ListOfMinutiae* resPool = new ListOfMinutiae;

	for (int i = 0; i < defaultThreadCount; i++)
	{
		while (pools[i]->head != NULL)
		{
			resPool->Add(pools[i]->Pop());
		}
	}

	return resPool;
}

int CountOfMinutiaes(int* counts, int length)
{
	int count = 0;

	for (int i = 0; i < length; i++)
	{
		printf("%d ", counts[i]);
		count += counts[i];
	}

	return count;
}

bool* Start(float* source, int step, int lengthWings, int width, int height)
{
	int sizeSection = lengthWings * 2 + 1;

	CUDAArray<float> image = CUDAArray<float>(source, width, height);

	dim3 blockSize = 1;
	dim3 gridSize = dim3(ceilMod(image.Height, defaultThreadCount), ceilMod(image.Width, defaultThreadCount));

	CUDAArray<float> orientationField = CUDAArray<float>(OrientationFieldInBlocks(source, width, height), height, width);
	CUDAArray<bool> visited = CUDAArray<bool>((bool*)calloc(width * height, sizeof(bool)), width, height);
	CUDAArray<int> countOfMinutiae = CUDAArray<int>((int*)calloc(gridSize.x * gridSize.y, sizeof(int)), gridSize.x * gridSize.y, 1);
	CUDAArray<Minutiae> minutiaes = CUDAArray<Minutiae>((Minutiae*)calloc(width * height, sizeof(Minutiae)), width * height, 1);
	//CUDAArray<ListOfMinutiae*> minutiaes = CUDAArray<ListOfMinutiae*>((ListOfMinutiae**)calloc(gridSize.x * gridSize.y, sizeof(ListOfMinutiae*)), gridSize.x * gridSize.y, 1);

	FindMinutia << <gridSize, blockSize >> > (image, orientationField, visited, countOfMinutiae, minutiaes, sizeSection, step);
	cudaDeviceSynchronize();
	cudaError_t e = cudaGetLastError(); 
	if (e != cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));
		exit(0);
	}

	int* counts = countOfMinutiae.GetData();

	for (int i = 0; i < gridSize.x * gridSize.y; i++)
	{
		printf("%d ", counts[i]);
	}

	//CountOfMinutiaes(countOfMinutiae.GetData(), gridSize.x * gridSize.y);

	return visited.GetData();

	/*ListOfMinutiae** notProcessedPools = minutiaes.GetData();

	return Parsing(MergeMinutiaePools(notProcessedPools));*/
}

int main(int argc, char *argv[])
{
	int width;
	int height;
	/*if (argc != 2)
	{
		printf("Need path to file");
		return 0;
	}*/
	char* filename = "H:\\GitHub\\CUDA-Fingerprinting\\Code\\CUDAFingerprinting.GPU.RidgeLine\\res.bmp";  //Write your way to bmp file
	int* img = loadBmp(filename, &width, &height);
	float* source = (float*)malloc(height*width*sizeof(float));
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			source[i * width + j] = (float)img[i * width + j];
		}

	bool* res = Start(source, 2, 3, width, height);
	
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			img[i * width + j] = res[i * width + j] ? 255 : 0;
		}



	saveBmp("..\\rez.bmp", img, width, height);

 	return 0;
}
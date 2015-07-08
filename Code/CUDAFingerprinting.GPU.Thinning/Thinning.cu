#include "Thinning.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdlib.h>
#include "ThinnerUtils.h"

//patterns
PixelType*** p = NULL;

double** a = NULL;
int w = 0;
int h = 0;

void init(double** arr, int width, int height)
{
	p = (PixelType***)malloc(sizeof(PixelType**) * 14);
	for (int i = 0; i < 14; i++)
	{
		p[i] = (PixelType**)malloc(sizeof(PixelType*) * 3);
		for (int j = 0; j < 3; j++)
		{
			p[i][j] = (PixelType*)malloc(sizeof(PixelType) * 3);
		}
	}
	//a
	p[0][0][0] = PixelType::FILLED; p[0][0][1] = PixelType::FILLED; p[0][0][2] = PixelType::AT_LEAST_ONE_EMPTY;
	p[0][1][0] = PixelType::FILLED; p[0][1][1] = PixelType::CENTER; p[0][1][2] = PixelType::EMPTY;
	p[0][2][0] = PixelType::FILLED; p[0][2][1] = PixelType::FILLED; p[0][2][2] = PixelType::AT_LEAST_ONE_EMPTY;
	//b
	p[1][0][0] = PixelType::FILLED;			    p[1][0][1] = PixelType::FILLED; p[1][0][2] = PixelType::FILLED;
	p[1][1][0] = PixelType::FILLED;			    p[1][1][1] = PixelType::CENTER; p[1][1][2] = PixelType::FILLED;
	p[1][2][0] = PixelType::AT_LEAST_ONE_EMPTY; p[1][2][1] = PixelType::EMPTY;  p[1][2][2] = PixelType::AT_LEAST_ONE_EMPTY;
	//c - needs special processing
	p[2][0][0] = PixelType::AT_LEAST_ONE_EMPTY; p[2][0][1] = PixelType::FILLED; p[2][0][2] = PixelType::FILLED;
	p[2][1][0] = PixelType::EMPTY;				p[2][1][1] = PixelType::CENTER; p[2][1][2] = PixelType::FILLED;//PixelType.FILLED 
	p[2][2][0] = PixelType::AT_LEAST_ONE_EMPTY; p[2][2][1] = PixelType::FILLED; p[2][2][2] = PixelType::FILLED;
	//d - needs special processing
	p[3][0][0] = PixelType::AT_LEAST_ONE_EMPTY; p[3][0][1] = PixelType::EMPTY;  p[3][0][2] = PixelType::AT_LEAST_ONE_EMPTY;
	p[3][1][0] = PixelType::FILLED;			    p[3][1][1] = PixelType::CENTER; p[3][1][2] = PixelType::FILLED;
	p[3][2][0] = PixelType::FILLED;			    p[3][2][1] = PixelType::FILLED; p[3][2][2] = PixelType::FILLED;
													    	//PixelType.FILLED
	//e
	p[4][0][0] = PixelType::ANY;	p[4][0][1] = PixelType::EMPTY;  p[4][0][2] = PixelType::EMPTY;
	p[4][1][0] = PixelType::FILLED;	p[4][1][1] = PixelType::CENTER; p[4][1][2] = PixelType::EMPTY;
	p[4][2][0] = PixelType::ANY;    p[4][2][1] = PixelType::FILLED; p[4][2][2] = PixelType::ANY;
	//f
	p[5][0][0] = PixelType::ANY;   p[5][0][1] = PixelType::FILLED; p[5][0][2] = PixelType::FILLED;
	p[5][1][0] = PixelType::EMPTY; p[5][1][1] = PixelType::CENTER; p[5][1][2] = PixelType::FILLED;
	p[5][2][0] = PixelType::EMPTY; p[5][2][1] = PixelType::EMPTY;  p[5][2][2] = PixelType::ANY;
	//g
	p[6][0][0] = PixelType::EMPTY; p[6][0][1] = PixelType::FILLED; p[6][0][2] = PixelType::EMPTY;
	p[6][1][0] = PixelType::EMPTY; p[6][1][1] = PixelType::CENTER; p[6][1][2] = PixelType::FILLED;
	p[6][2][0] = PixelType::EMPTY; p[6][2][1] = PixelType::EMPTY;  p[6][2][2] = PixelType::EMPTY;
	//h
	p[7][0][0] = PixelType::ANY;    p[7][0][1] = PixelType::FILLED; p[7][0][2] = PixelType::ANY;
	p[7][1][0] = PixelType::FILLED; p[7][1][1] = PixelType::CENTER; p[7][1][2] = PixelType::EMPTY;
	p[7][2][0] = PixelType::ANY;    p[7][2][1] = PixelType::EMPTY;  p[7][2][2] = PixelType::EMPTY;
	//i
	p[8][0][0] = PixelType::EMPTY; p[8][0][1] = PixelType::EMPTY;  p[8][0][2] = PixelType::ANY;
	p[8][1][0] = PixelType::EMPTY; p[8][1][1] = PixelType::CENTER; p[8][1][2] = PixelType::FILLED;
	p[8][2][0] = PixelType::ANY;   p[8][2][1] = PixelType::FILLED; p[8][2][2] = PixelType::FILLED;
	//j
	p[9][0][0] = PixelType::EMPTY; p[9][0][1] = PixelType::EMPTY;  p[9][0][2] = PixelType::EMPTY;
	p[9][1][0] = PixelType::EMPTY; p[9][1][1] = PixelType::CENTER; p[9][1][2] = PixelType::FILLED;
	p[9][2][0] = PixelType::EMPTY; p[9][2][1] = PixelType::FILLED; p[9][2][2] = PixelType::EMPTY;
	//k
	p[10][0][0] = PixelType::EMPTY;  p[10][0][1] = PixelType::EMPTY;  p[10][0][2] = PixelType::EMPTY;
	p[10][1][0] = PixelType::EMPTY;  p[10][1][1] = PixelType::CENTER; p[10][1][2] = PixelType::EMPTY;
	p[10][2][0] = PixelType::FILLED; p[10][2][1] = PixelType::FILLED; p[10][2][2] = PixelType::FILLED;
	//l
	p[11][0][0] = PixelType::FILLED; p[11][0][1] = PixelType::EMPTY;  p[11][0][2] = PixelType::EMPTY;
	p[11][1][0] = PixelType::FILLED; p[11][1][1] = PixelType::CENTER; p[11][1][2] = PixelType::EMPTY;
	p[11][2][0] = PixelType::FILLED; p[11][2][1] = PixelType::EMPTY;  p[11][2][2] = PixelType::EMPTY;
	//m
	p[12][0][0] = PixelType::FILLED; p[12][0][1] = PixelType::FILLED; p[12][0][2] = PixelType::FILLED;
	p[12][1][0] = PixelType::EMPTY;  p[12][1][1] = PixelType::CENTER; p[12][1][2] = PixelType::EMPTY;
	p[12][2][0] = PixelType::EMPTY;  p[12][2][1] = PixelType::EMPTY;  p[12][2][2] = PixelType::EMPTY;
	//n
	p[13][0][0] = PixelType::EMPTY; p[13][0][1] = PixelType::EMPTY;  p[13][0][2] = PixelType::FILLED;
	p[13][1][0] = PixelType::EMPTY; p[13][1][1] = PixelType::CENTER; p[13][1][2] = PixelType::FILLED;
	p[13][2][0] = PixelType::EMPTY; p[13][2][1] = PixelType::EMPTY;  p[13][2][2] = PixelType::FILLED;
	a = copy2DArray(arr, width, height);
	w = width;
	h = height;
}

double GetPixel(double** array, int x, int y)
{
	return (x < 0 || y < 0 || x >= w || y >= h) ?
		WHITE :
		array[h - 1 - y][x] > 128.0 ?
			WHITE :
			BLACK;
}

void SetPixel(double** array, int x, int y, double value)
{
	if (x < 0 || y < 0 || x >= w || y >= h) return;
	array[h - 1 - y][x] = value;
}

bool AreEqual(double value, PixelType patternPixel)
{
	switch (patternPixel)
	{
	case PixelType::FILLED:
	{
		if (value == BLACK)
			return true;
		break;
	}
	case PixelType::EMPTY:
	{
		if (value == WHITE)
			return true;
		break;
	}
	case PixelType::AT_LEAST_ONE_EMPTY://y
		return true;
	case PixelType::CENTER://c
		if (value == BLACK)
			return true;
		break;
	case PixelType::ANY://x
		return true;
	default:
		break;
	}
	return false;
}

//-1 - no match
int MatchPattern(int x, int y)
{
	if (GetPixel(a, x, y) == WHITE) return -1;
	for (int i = 0; i < 14; i++)
	{
		bool yInPattern = false;
		int yCounter = 0;
		bool bad = false;
		for (int dX = -1; dX < 2; dX++)
		{
			if (bad)
				break;
			for (int dY = -1; dY < 2; dY++)
			{
				if (p[i][1 + dX][1 + dY] == PixelType::AT_LEAST_ONE_EMPTY)
				{
					yInPattern = true;
					yCounter += GetPixel(a, x + dX, y + dY) == WHITE ? 1 : 0;
					continue;
				}
				if (!AreEqual(GetPixel(a, x + dX, y + dY), p[i][1 + dX][1 + dY]))
				{
					bad = true;
					break;
				}
			}
		}
		if (bad)
			continue;
		if (yInPattern && yCounter == 0)
			continue;
		if (i == 2 && !AreEqual(GetPixel(a, x, y + 2), PixelType::FILLED))
		{
			continue;
		}
		else if (i == 3 && !AreEqual(GetPixel(a, x + 2, y), PixelType::FILLED))
		{
			continue;
		}
		return i;
	}
	return -1;
}

double** Thin(double** arr, int width, int height)
{
	init(arr, width, height);
	bool isSkeleton;
	double** buffer = copy2DArray(a, width, height);
	do
	{
		isSkeleton = true;
		for (int y = 0; y < h; y++)
		{
			for (int x = 0; x < w; x++)
			{
				int pattern = MatchPattern(x, y);
				if (pattern != -1)
				{
					SetPixel(buffer, x, y, WHITE);
					isSkeleton = false;
				}
			}
		}
		free2DArray(a, width, height);
		a = copy2DArray(buffer, width, height);
	} while (!isSkeleton);
	free2DArray(a, width, height);
	for (int i = 0; i < 14; i++)
	{
		free2DArray(p[i], 3, 3);
	}
	free(p);
	return buffer;
}
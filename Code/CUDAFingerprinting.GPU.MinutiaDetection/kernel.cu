#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <memory.h>
#include <math.h>

extern "C"
{
	__declspec(dllexport) int GetMinutias(float* dest, int* data, double* orientation, int width, int height);
}

const int BLACK = 0;
const int GREY = 128;
const int WHITE = 255;

int w = 0;
int h = 0;

double GetPixel(int* data, int x, int y)
{
	return (x < 0 || y < 0 || x >= w || y >= h) ?
	WHITE :
		  data[(h - 1 - y) * w + x] > GREY ?
	  WHITE :
			BLACK;
}

int NeigboursCount;

bool IsMinutia(int* data, int x, int y)
{
	if (GetPixel(data, x, y) != BLACK)
		return false;
	//check 8-neigbourhood
	bool p[8] = {
		GetPixel(data, x, y - 1) > 0,
		GetPixel(data, x + 1, y - 1) > 0,
		GetPixel(data, x + 1, y) > 0,
		GetPixel(data, x + 1, y + 1) > 0,
		GetPixel(data, x, y + 1) > 0,
		GetPixel(data, x - 1, y + 1) > 0,
		GetPixel(data, x - 1, y) > 0,
		GetPixel(data, x - 1, y - 1) > 0,
	};

	NeigboursCount = 0;
	for (int i = 1; i < 9; i++)
	{
		NeigboursCount += p[i % 8] ^ p[i - 1] ? 1 : 0;
	}
	NeigboursCount /= 2;
	//count == 0 <=> isolated point - NOT minutia
	//count == 1 <=> 'end line' - minutia
	//count == 2 <=> part of the line - NOT minutia
	//count == 3 <=> 'fork' - minutia
	//count >= 3 <=> composit minutia - ignoring in this implementation
	return ((NeigboursCount == 1) || (NeigboursCount == 3));
}

bool InCircle(int xC, int yC, int R, int x, int y)
{
	return pow((double)(xC - x), 2) + pow((double)(yC - y), 2) < R * R;
}

double GetCorrectAngle(int* data, double* orientation, int x, int y)
{
	double angle = orientation[(h - 1 - y) * w + x];
	float PI = 3.141592654f;
	//for 'end line' minutia
	if (NeigboursCount == 1)
	{
		if (angle > 0.0)
		{
			if ((GetPixel(data, x, y - 1) +
				GetPixel(data, x + 1, y - 1) +
				GetPixel(data, x + 1, y))
				<
				(GetPixel(data, x, y + 1) +
				GetPixel(data, x - 1, y + 1) +
				GetPixel(data, x - 1, y)))
			{
				angle += PI;
			}
		}
		else
		{
			if ((GetPixel(data, x, y + 1) +
				GetPixel(data, x + 1, y + 1) +
				GetPixel(data, x + 1, y))
				<
				(GetPixel(data, x, y - 1) +
				GetPixel(data, x - 1, y - 1) +
				GetPixel(data, x - 1, y)))
			{
				angle += PI;
			}
		}
	}
	//for 'fork' minutia
	else if (NeigboursCount == 3)
	{
		for (int r = 1; r < 16; r++)
		{
			double normal = angle + PI / 2;
			int aboveNormal = 0;
			int belowNormal = 0;

			for (int i = -r; i <= r; i++)
			{
				for (int j = -r; j <= r; j++)
				{
					if (i == j && j == 0)
					{
						continue;
					}
					if (GetPixel(data, x + j, y + i) == BLACK &&
						InCircle(x, y, r, x + j, y + i))
					{
						double deltaNormalY = - tan(normal) * j;
						if (i < deltaNormalY)
						{
							aboveNormal++;
						}
						else
						{
							belowNormal++;
						}
					}
				}
			}
			if (aboveNormal == belowNormal)
			{
				continue;//?
			}
			else
			{
				if ((aboveNormal > belowNormal &&
					tan(angle) > 0.0) ||
					(aboveNormal < belowNormal &&
					tan(angle) < 0.0))
				{
					angle += PI;
				}
				break;
			}
		}
	}
	return angle;
}

//returns number of found minutias
//in result:
//dest[i * 3 + 0] - x coord of i's minutia
//dest[i * 3 + 1] - y coord of i's minutia
//dest[i * 3 + 2] - direction of i's minutia
int GetMinutias(float* dest, int* data, double* orientation, int width, int height)
{
	w = width;
	h = height;
	int minutiasCount = 0;
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			if (IsMinutia(data, x, y))
			{
				dest[minutiasCount * 3] = (float)x;
				dest[minutiasCount * 3 + 1] = (float)y;
				dest[minutiasCount * 3 + 2] = (float)GetCorrectAngle(
					data,
					orientation,
					x,
					y
				);
				minutiasCount++;
			}
		}
	}
	return minutiasCount;
}
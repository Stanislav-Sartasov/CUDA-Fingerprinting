#include "kernel.cuh"
#include <stdlib.h>

int* _x;
int* _y;
int* _mType;
float* _angle;

__host__
bool Parsing(ListOfMinutiae* minutiaeList)
{
	int count = 0;

	_x = (int*)malloc(sizeof(int)*count);
	_y = (int*)malloc(sizeof(int)*count);
	_mType = (int*)malloc(sizeof(int)*count);
	_angle = (float*)malloc(sizeof(float)*count);

	int i = 0;

	while (minutiaeList->head != NULL)
	{
		Minutiae foo = minutiaeList->Pop();

		//if (foo == NULL) return false;

		_x[i] = foo.x;
		_y[i] = foo.y;
		_mType[i] = foo.type;
		_angle[i] = foo.angle;

		i++;
	}

	return true;
}


int* GetX()
{
	return _x;
}

int* GetY()
{
	return _y;
}

int* GetMType()
{
	return _mType;
}

float* GetAngle()
{
	return _angle;
}
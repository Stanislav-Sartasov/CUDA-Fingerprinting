#include "cuda_runtime.h"
#include <iostream>
#include "device_launch_parameters.h"
#include "device_functions.h"
#include <stdio.h>
#include "constsmacros.h"
#include <stdlib.h>
#include <math.h>
#include "ImageLoading.cuh"
#include "CUDAArray.cuh"
#include <float.h>
#include "OrientationField.cuh"

enum Direction
{
	Forward,
	Back
};

enum MinutiaeType
{
	NotMinutiae,
	LineEnding,
	Intersection
};

struct Point
{
	int x;
	int y;
};

struct Minutiae
{
	int x;
	int y;
	float angle;
	MinutiaeType type;
};

struct Queue
{
	struct Node
	{
		Point point;
		Node* next;
	};

	Node* head;
	Node* tail;
	int count;

	public:

		__device__ __host__
		Queue()
		{
			head = NULL;
			tail = NULL;
			count = 0;
		}

		__device__ __host__
		void Push(Point point)
		{
			Node* newElem = new Node;
			newElem->point = point;
			newElem->next = NULL;

			if (head != NULL)
			{
				head = newElem;
				tail = head;
			}
			else
			{
				tail->next = newElem;
				tail = newElem;
			}

			count++; 
		}

		__device__ __host__
		Point Pop()
		{
			Point point = head->point;
			Node* next = head->next;

			delete(head);
			head = next;
			count--;

			if (head == NULL) tail = NULL;

			return point;
		}

		~Queue()
		{
			while (head != NULL)
			{
				Node* n = head->next;
				delete head;
				head = n;
			}
		}
};

struct ListOfMinutiae
{
	struct Node
	{
		Minutiae minutiae;
		Node* next;
	};
	
	Node* head;
	Node* tail;

	public:
		__device__ __host__
		ListOfMinutiae()
		{
			head = NULL;
			tail = NULL;
		}

		~ListOfMinutiae()
		{
			while (head != NULL)
			{
				Node* n = head->next;
				delete head;
				head = n;
			}
		}

		__device__ __host__
		void Add(Minutiae newMinutiae)
		{
			Node* newElem = new Node;
			newElem->minutiae = newMinutiae;
			newElem->next = NULL;

			if (head != NULL)
			{
				head = newElem;
				tail = head;
			}
			else
			{
				tail->next = newElem;
				tail = newElem;
			}
		}

		
};

__global__ void FindMinutia(CUDAArray<float> image, CUDAArray<float> orientationField, CUDAArray<bool> visited,
	CUDAArray<int> countOfMinutiae, CUDAArray<ListOfMinutiae*> minutiaes,
	const int size, const int step, int colorThreshold);
#include "cuda_runtime.h"
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
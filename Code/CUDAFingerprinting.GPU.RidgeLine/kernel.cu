#include "kernel.cuh"
#define M_PI 3.14159265358979323846

class RidgeOnLine
{
private:
	const double Pi4 = M_PI / 4;
	int _countTest;
	Direction _direction;
	float* _image;
	int _lengthWings;

	ListOfMinutiae* _minutiaes;
	float* _orientationField;

	Point* _sectionIndexes;
	int _step;
	bool* _visited;

public:
	RidgeOnLine(float* image, int step, int lengthWings, int width, int height)
	{
		_countTest = 0;
		
		_image = image;
		_orientationField = OrientationFieldInBlocks(_image, width, height);
		_step = step;
		_lengthWings = lengthWings;
		_sectionIndexes = (Point*)malloc(sizeof(Point) * (lengthWings + 1));
		_minutiaes = new ListOfMinutiae;
		_visited = (bool*)malloc(sizeof(bool) * width * height);
	}

	~RidgeOnLine()
	{
		delete _sectionIndexes;
		delete _visited;
		delete _minutiaes;
	}


};
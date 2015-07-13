#ifndef CUDAFINGEROPRINTING_THINNERUTILS
#define CUDAFINGEROPRINTING_THINNERUTILS

#ifndef BLACK
#define BLACK 0.0
#endif

#ifndef WHITE
#define WHITE 255.0
#endif

enum PixelType
{
	FILLED,
	EMPTY,
	ANY,
	CENTER,
	AT_LEAST_ONE_EMPTY
};

double* copy1DArray(double* source, int size);

//double** copy2DArray(double** source, int width, int height);

double* copy2DArrayTo1D(double** source, int width, int height);

double** copy1DArrayTo2D(double* source, int width, int height);

#endif
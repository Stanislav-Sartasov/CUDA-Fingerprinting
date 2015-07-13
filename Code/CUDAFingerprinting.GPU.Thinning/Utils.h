#ifndef CUDAFINGEROPRINTING_UTILS
#define CUDAFINGEROPRINTING_UTILS

/*Test helpers*/

double** intToDoubleArray(int* input, int width, int height);

int* doubleToIntArray(double** input, int width, int height);

int* doubleToIntArray(double* input, int width, int height);

//void WriteArrayPic(double** bytes, int h, int w);

//overlaps skeleton above background
double** OverlapArrays(double** skeleton, double** background, int width, int height);

int* OverlapArrays(int* skeleton, int* background, int width, int height);

#endif
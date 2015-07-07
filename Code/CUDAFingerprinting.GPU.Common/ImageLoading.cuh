#ifndef CUDAFINGEROPRINTING_IMAGELOADING
#define CUDAFINGEROPRINTING_IMAGELOADING

int* loadBmp(char* filename, int* width, int* height);

void saveBmp(char* filename, int* data, int width, int height);

#endif
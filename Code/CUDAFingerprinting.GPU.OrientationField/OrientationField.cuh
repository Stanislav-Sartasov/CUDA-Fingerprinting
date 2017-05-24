#ifndef CUDAFINGERPRINTING_ORIENTATION
#define CUDAFINGERPRINTING_ORIENTATION

extern "C"
{
	__declspec(dllexport) void OrientationFieldInPixels(float* res, float* floatArray, int width, int height);
}

float* OrientationFieldInBlocks(float* floatArray, int width, int height);

float* OrientationFieldInPixels(float* floatArray, int width, int height);
void OrientationFieldInPixels(float* res, float* floatArray, int width, int height);

#endif
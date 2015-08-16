#ifndef CUDAFINGERPRINTING_ORIENTATIONREGULARIZATION
#define CUDAFINGERPRINTING_ORIENTATIONREGULARIZATION

extern "C"
{
	__declspec(dllexport) void OrientationRegularizationPixels(float* res, float* floatArray, int height, int width);
}
float* OrientationRegularizationPixels(float* floatArray, int height, int width);
void OrientationRegularizationPixels(float* res, float* floatArray, int height, int width);

#endif
#ifndef CUDAFINGERPRINTING_ORIENTATIONREGULARIZATION
#define CUDAFINGERPRINTING_ORIENTATIONREGULARIZATION

extern "C"
{
	__declspec(dllexport) void OrientationRegularizationPixels(float* res, float* floatArray, int height, int width, int sizeFil);
}
float* OrientationRegularizationPixels(float* floatArray, int height, int width, int sizeFil);
void OrientationRegularizationPixels(float* res, float* floatArray, int height, int width, int sizeFil);

#endif
#ifndef CUDAFINGEROPRINTING_ENHANCE
#define CUDAFINGEROPRINTING_ENHANCE
extern "C"
{
	__declspec(dllexport) void Enhance(float* source, int imgWidth, int imgHeight, float* res, float* orientationMatrix,
		float* frequencyMatr, int filterSize, int angleNum);
	__declspec(dllexport) void Enhance16(float* source, int imgWidth, int imgHeight, float* res, float* orientationMatrix,
		float* frequencyMatr, int angleNum);
	__declspec(dllexport) void Enhance32(float* source, int imgWidth, int imgHeight, float* res, float* orientationMatrix,
		float* frequencyMatr, int angleNum);
}
void Enhance(float* source, int imgWidth, int imgHeight, float* res, float* orientationMatrix,
	float* frequencyMatr, int filterSize, int angleNum);
#endif
#ifndef CUDAFINGEROPRINTING_FILTER
#define CUDAFINGEROPRINTING_FILTER

extern "C"
{
	__declspec(dllexport) CUDAArray<float> MakeGabor16Filters(int angleNum, float frequency);
	__declspec(dllexport) CUDAArray<float> MakeGabor32Filters(int angleNum, float frequency);
}

#endif
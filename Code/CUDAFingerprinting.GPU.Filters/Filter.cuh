#ifndef CUDAFINGEROPRINTING_FILTER
#define CUDAFINGEROPRINTING_FILTER

extern "C"
{
	__declspec(dllexport) void MakeGabor16Filters(float* filter, int angleNum, float frequency);
	__declspec(dllexport) void MakeGabor32Filters(float* filter, int angleNum, float frequency);
}

#endif
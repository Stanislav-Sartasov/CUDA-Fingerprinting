#ifndef CUDAFINGERPRINTING_MINUTIASTRUCTS
#define CUDAFINGERPRINTING_MINUTIASTRUCTS


struct Minutia
{
	float angle;
	int x;
	int y;
};

struct Descriptor
{
	Minutia minutias[128];
	Minutia center;
	int length;
};

struct Tuple
{
	float value;
	int idx1;
	int idx2;
};

#endif
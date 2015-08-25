#ifndef LOOK_UP_TABLE
#define LOOK_UP_TABLE

#include "cuda_runtime.h"

static texture<short> texLUTSource;

// Look-up table
template<class T>
class LUT {
public:
	T *cudaPtr;
	LUT() { cudaPtr = 0; }

	size_t getSize() const {
		return size;
	}

	bool canBeRead(const char fileName[]);

	virtual __device__ T operator[](int pos) = 0;

	virtual void free() = 0;

	virtual ~LUT() {}

protected:
	size_t size;
	
};

// Look-up table for number of local matching scores to be considered
class LUT_NP : public LUT<short> {
public:
	LUT_NP() {}

	void fill(int upperBound = MAX_CYLINDERS);

	// Text file must contain a line with the table upper bound
	void read(const char fileName[], size_t limit = MAX_CYLINDERS);

	__device__ short operator[](int pos);

	void free();

	~LUT_NP() {}

	const static short MAX_CYLINDERS, MIN_NP, MAX_NP;
	const static float TAU, MU;
};

// Look-up table for template tags
class LUT_C : public LUT<int> {
public:
	LUT_C() {}

	void fill(short templateSizes[], size_t templatesNumber);

	void read(const char fileName[]);

	__device__ int operator[](int pos);

	void free();

	~LUT_C() {}

	static float TAU, MU;
};

#endif
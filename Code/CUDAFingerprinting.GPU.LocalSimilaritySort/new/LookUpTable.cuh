#ifndef LOOK_UP_TABLE
#define LOOK_UP_TABLE

#include "cuda_runtime.h"

static texture<short> texLUTSource;

// Look-up table
template<class T>
class LUT {
public:

	LUT() { cudaPtr = 0; }

	size_t getLength() const {
		return size;
	}

	// bool canBeRead(const char fileName[]);

	virtual __device__ T operator[](size_t pos)  const = 0;

	virtual void free() = 0;

	virtual ~LUT() {}

protected:
	size_t size;
	T *cudaPtr;
};

// Look-up table for the number of local matching scores to be considered
class LUT_NP : public LUT<short> {
public:
	LUT_NP() {}

	void fill(short upperBound = MAX_CYLINDERS);

	// Text file must contain a line with the table upper bound
	int read(const char fileName[], size_t limit = MAX_CYLINDERS);

	__device__ short operator[](size_t pos) const;

	void free();

	~LUT_NP() {}

	const static short MAX_CYLINDERS, MIN_NP, MAX_NP;
	const static float TAU, MU;
};

// Look-up table for a template original row
class LUT_OR : public LUT<int> {
public:
	size_t databaseHeight;

	LUT_OR() {}

	void fill(size_t templatesNumber, short templateSizes[]);

	// Text file must contain a line with the number of templates in database and the size of each template
	int read(const char fileName[]);

	__device__ int operator[](size_t pos) const;

	void free();

	~LUT_OR() {}

	static float TAU, MU;
};

#endif
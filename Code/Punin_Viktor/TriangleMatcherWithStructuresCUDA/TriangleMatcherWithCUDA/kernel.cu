#include "cuda_runtime.h"
#include "DEVICE_launch_parameters.h"

#include <malloc.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "TrianglesMatcherWithCUDA.cuh"

int main()
{
	int max_rand = 100;

	int ABC_size = 100;
	int ABCsize = 200;
	Triangle* ABC = (Triangle*)malloc(ABCsize * sizeof(Triangle));
	Triangle* ABC_ = (Triangle*)malloc(ABC_size * sizeof(Triangle));

	srand(time(NULL));                      

	for (int i = 0; i < ABCsize; i++)
	{
		Triangle ABCt;

		ABCt.A.x = max_rand / 2 - rand() % max_rand;
		ABCt.A.y = max_rand / 2 - rand() % max_rand;

		ABCt.B.x = max_rand / 2 - rand() % max_rand;
		ABCt.B.y = max_rand / 2 - rand() % max_rand;

		ABCt.C.x = max_rand / 2 - rand() % max_rand;
		ABCt.C.y = max_rand / 2 - rand() % max_rand;
		ABC[i] = ABCt;
	}

	for (int i = 0; i < ABC_size; i++)
	{
		Triangle ABCt;

		ABCt.A.x = max_rand / 2 - rand() % max_rand;
		ABCt.A.y = max_rand / 2 - rand() % max_rand;

		ABCt.B.x = max_rand / 2 - rand() % max_rand;
		ABCt.B.y = max_rand / 2 - rand() % max_rand;

		ABCt.C.x = max_rand / 2 - rand() % max_rand;
		ABCt.C.y = max_rand / 2 - rand() % max_rand;
		ABC_[i] = ABCt;
	}


	TransformationWithDistance* result = (TransformationWithDistance*)malloc(ABC_size * ABCsize * sizeof(TransformationWithDistance));
	
	Transformation* resultTransformation = (Transformation*)malloc(ABC_size * ABCsize * sizeof(Transformation));
	float* resultDistance = (float*)malloc(ABC_size * ABCsize * sizeof(float));
	cudaError_t cudaStatus = findOptimumTransformationPlusDistanceVersionCUDA(ABC_, ABC_size, ABC, ABCsize, resultTransformation, resultDistance, 10, 0.00001f, 5);
	
	Transformation zeroResult = resultTransformation[56 * 200 + 4];
	float zeroDistance = resultDistance[56 * 200 + 4];

	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = findOptimumTransformationParallelForWithCuda(ABC_, ABC_size, ABC, ABCsize, result, 10, 0.00001f, 5);
	TransformationWithDistance firstResult = result[56 * 200 + 4];
	
	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		goto End;

	cudaStatus = findOptimumTransformationNonParallelForWithCuda(ABC_, ABC_size, ABC, ABCsize, result, 10, 0.00001f, 5);
	TransformationWithDistance secondResult = result[56 * 200 + 4];
	
	if (cudaStatus != cudaSuccess)
		goto End;
		
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
		goto End;

	float distance = countDistanceBetweenTriangles(&ABC_[56], &ABC[4]);

End:
	free(ABC);
	free(ABC_);
	free(result);

	return 0;
}
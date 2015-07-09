#include "d:/Git/CUDA-Fingerprinting/Code/CUDAFingerprinting.GPU.Common/ImageLoading.cuh"
#include "d:/Git/CUDA-Fingerprinting/Code/CUDAFingerprinting.GPU.Common/CUDAArray.cuh"
#include "d:/Git/CUDA-Fingerprinting/Code/CUDAFingerprinting.GPU.Common/Convolution.cuh"

template<typename T> class CUDASegmentator
{
private:
	int* imgPtr; //Bitmap pic; //Saves a source picture
	T **matrix; //Saves results of using Sobel filter
public:
	CUDASegmentator ();

	void SobelFilter (char* filename, int Width, int Height);

	void MatrixMaking (char* filename, int Width, int Height);

	void BWPicture();

	~CUDASegmentator();
};

template class CUDASegmentator<int>;

template class CUDASegmentator<double>;
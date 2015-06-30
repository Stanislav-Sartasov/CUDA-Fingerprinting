#ifndef CUDAFINGEROPRINTING_IMAGELOADING
#define CUDAFINGEROPRINTING_IMAGELOADING

typedef struct
{
	unsigned short    BfType;
	unsigned long   BfSize;
	unsigned short    BfReserved1;
	unsigned short    BfReserved2;
	unsigned long   BfOffBits;
	unsigned long      BiSize;
	long       Width;
	long       Height;
	unsigned short       BiPlanes;
	unsigned short       BiBitCount;
	unsigned long      BiCompression;
	unsigned long      BiSizeImage;
	long       BiXPelsPerMeter;
	long       BiYPelsPerMeter;
	unsigned long      BiClrUsed;
	unsigned long      BiClrImportant;
} BMPHeader;

int* loadBmp(BMPHeader* header, char* filename);

void saveBmp(int* data, BMPHeader* header, char* filename);

#endif
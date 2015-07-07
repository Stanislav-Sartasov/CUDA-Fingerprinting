#include "ImageLoading.cuh"
#include <stdio.h>
#include <stdlib.h>

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

typedef struct
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char A;
} RGBAPixel;

typedef struct
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
} RGBPixel;

int* loadBmp(char* filename, int* width, int* height)
{
	FILE *input = fopen(filename, "rb");
	
	BMPHeader header;

	fread(&header.BfType, 2, 1, input);
	fread(&header.BfSize, 4, 1, input);
	fread(&header.BfReserved1, 2, 1, input);
	fread(&header.BfReserved2, 2, 1, input);
	fread(&header.BfOffBits, 4, 1, input);
	fread(&header.BiSize, 4, 1, input);
	fread(&header.Width, 4, 1, input);
	fread(&header.Height, 4, 1, input);
	fread(&header.BiPlanes, 2, 1, input);
	fread(&header.BiBitCount, 2, 1, input);
	fread(&header.BiCompression, 4, 1, input);
	fread(&header.BiSizeImage, 4, 1, input);
	fread(&header.BiXPelsPerMeter, 4, 1, input);
	fread(&header.BiYPelsPerMeter, 4, 1, input);
	fread(&header.BiClrUsed, 4, 1, input);
	fread(&header.BiClrImportant, 4, 1, input);

	*width = header.Width;
	*height = header.Height;
	
	if (header.BiBitCount != 32 && header.BiBitCount != 24)
	{
		return NULL;
	}

	int* transformedPixels = (int*)malloc(sizeof(int)* header.Width * header.Height);

	if (header.BiBitCount == 32)
	{
		RGBAPixel* rawImage = (RGBAPixel*)malloc(sizeof(RGBAPixel)* header.Width * header.Height);

		size_t size = fread(rawImage, sizeof(RGBAPixel), header.Width * header.Height, input);

		for (int row = 0; row < header.Height; row++)
		{
			for (int column = 0; column < header.Width; column++)
			{
				transformedPixels[(header.Height - 1 - row) * header.Width + column] = rawImage[row * header.Width + column].R;
			}
		}

		free(rawImage);
	}
	else // 24
	{
		RGBPixel* rawImage = (RGBPixel*)malloc(sizeof(RGBPixel)* header.Width * header.Height);

		size_t size = fread(rawImage, sizeof(RGBPixel), header.Width * header.Height, input);

		for (int row = 0; row < header.Height; row++)
		{
			for (int column = 0; column < header.Width; column++)
			{
				transformedPixels[(header.Height - 1 - row) * header.Width + column] = rawImage[row * header.Width + column].R;
			}
		}

		free(rawImage);
	}

	fclose(input);

	return transformedPixels;

	return transformedPixels;
}

void saveBmp(char* filename, float* data, int width, int height)
{
	FILE *output;

	output = fopen(filename, "wb");

	BMPHeader header;
	header.BfType = 0x4D42; // due to the way "BM" being loaded as short integer, thus reverting the bytes order

	header.BfSize = 54 + sizeof(RGBAPixel) * width * height;

	header.BfReserved1 = header.BfReserved2 = 0;

	header.BfOffBits = 54;

	header.BiSize = 40; // size of BITMAPINFOHEADER is 40 bytes

	header.Width = width;
	header.Height = height;

	header.BiPlanes = 1; // constant

	header.BiBitCount = 32; // right now supporting only 32 bits per pixel

	header.BiCompression = 0; // no compression

	header.BiSizeImage = 0; // dummy, it's ARGB image without compression

	header.BiXPelsPerMeter = header.BiYPelsPerMeter = 3780;

	header.BiClrUsed = 0; // no palette used
	header.BiClrImportant = 0;

	fwrite(&header.BfType, 2, 1, output);
	fwrite(&header.BfSize, 4, 1, output);
	fwrite(&header.BfReserved1, 2, 1, output);
	fwrite(&header.BfReserved2, 2, 1, output);
	fwrite(&header.BfOffBits, 4, 1, output);
	fwrite(&header.BiSize, 4, 1, output);
	fwrite(&header.Width, 4, 1, output);
	fwrite(&header.Height, 4, 1, output);
	fwrite(&header.BiPlanes, 2, 1, output);
	fwrite(&header.BiBitCount, 2, 1, output);
	fwrite(&header.BiCompression, 4, 1, output);
	fwrite(&header.BiSizeImage, 4, 1, output);
	fwrite(&header.BiXPelsPerMeter, 4, 1, output);
	fwrite(&header.BiYPelsPerMeter, 4, 1, output);
	fwrite(&header.BiClrUsed, 4, 1, output);
	fwrite(&header.BiClrImportant, 4, 1, output);

	RGBAPixel* rawImage = (RGBAPixel*)malloc(sizeof(RGBAPixel)* width * height);

	for (int row = 0; row < height; row++)
	{
		for (int column = 0; column < width; column++)
		{
			rawImage[row*width + column].A = 255;
			rawImage[row*width + column].R = rawImage[row*width + column].G = rawImage[row*width + column].B =
				(int) data[(height - 1 - row)*width + column];
		}
	}

	fwrite(rawImage, sizeof(RGBAPixel), width * height, output);

	free(rawImage);

	fclose(output);
}
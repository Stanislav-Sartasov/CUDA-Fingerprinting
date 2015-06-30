#include "ImageLoading.cuh"
#include <stdio.h>
#include <stdlib.h>

typedef struct
{
	unsigned char B;
	unsigned char G;
	unsigned char R;
	unsigned char A;
} Pixel;

int* loadBmp(BMPHeader* header, char* filename)
{
	FILE *input = fopen(filename, "rb");
	
	fread(&header->BfType, 2, 1, input);
	fread(&header->BfSize, 4, 1, input);
	fread(&header->BfReserved1, 2, 1, input);
	fread(&header->BfReserved2, 2, 1, input);
	fread(&header->BfOffBits, 4, 1, input);
	fread(&header->BiSize, 4, 1, input);
	fread(&header->Width, 4, 1, input);
	fread(&header->Height, 4, 1, input);
	fread(&header->BiPlanes, 2, 1, input);
	fread(&header->BiBitCount, 2, 1, input);
	fread(&header->BiCompression, 4, 1, input);
	fread(&header->BiSizeImage, 4, 1, input);
	fread(&header->BiXPelsPerMeter, 4, 1, input);
	fread(&header->BiYPelsPerMeter, 4, 1, input);
	fread(&header->BiClrUsed, 4, 1, input);
	fread(&header->BiClrImportant, 4, 1, input);
	
	Pixel* rawImage = (Pixel*)malloc(sizeof(Pixel)* header->Width * header->Height);

	size_t size = fread(rawImage, sizeof(Pixel), header->Width * header->Height, input);


	int* transformedPixels = (int*)malloc(sizeof(int)* header->Width * header->Height);

	for (int row = 0; row < header->Height; row++)
	{
		for (int column = 0; column < header->Width; column++)
		{
			transformedPixels[(header->Height -1 -row)*header->Width + column] = rawImage[row*header->Width + column].R;
		}
	}

	free(rawImage);

	fclose(input);

	return transformedPixels;
}

void saveBmp(int* data, BMPHeader* header, char* filename)
{
	FILE *output;

	output = fopen(filename, "wb");

	fwrite(&header->BfType, 2, 1, output);
	fwrite(&header->BfSize, 4, 1, output);
	fwrite(&header->BfReserved1, 2, 1, output);
	fwrite(&header->BfReserved2, 2, 1, output);
	fwrite(&header->BfOffBits, 4, 1, output);
	fwrite(&header->BiSize, 4, 1, output);
	fwrite(&header->Width, 4, 1, output);
	fwrite(&header->Height, 4, 1, output);
	fwrite(&header->BiPlanes, 2, 1, output);
	fwrite(&header->BiBitCount, 2, 1, output);
	fwrite(&header->BiCompression, 4, 1, output);
	fwrite(&header->BiSizeImage, 4, 1, output);
	fwrite(&header->BiXPelsPerMeter, 4, 1, output);
	fwrite(&header->BiYPelsPerMeter, 4, 1, output);
	fwrite(&header->BiClrUsed, 4, 1, output);
	fwrite(&header->BiClrImportant, 4, 1, output);

	Pixel* rawImage = (Pixel*)malloc(sizeof(Pixel)* header->Width * header->Height);

	for (int row = 0; row < header->Height; row++)
	{
		for (int column = 0; column < header->Width; column++)
		{
			rawImage[row*header->Width + column].A = 255;
			rawImage[row*header->Width + column].R = rawImage[row*header->Width + column].G = rawImage[row*header->Width + column].B = 
				data[(header->Height - 1 - row)*header->Width + column];
		}
	}

	fwrite(rawImage, sizeof(Pixel), header->Width * header->Height, output);

	free(rawImage);

	fclose(output);
}
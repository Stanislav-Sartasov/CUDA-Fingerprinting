#include <iostream> 
#include <iomanip> 
#include <vector>
using namespace std;

typedef struct
{
	unsigned short  bfType;
	unsigned int    bfSize;
	unsigned short  bfReserved1;
	unsigned short  bfReserved2;
	unsigned int   bfOffBits;

	unsigned int    biSize;
	int             biWidth;
	int             biHeight;
	unsigned short  biPlanes;
	unsigned short  biBitCount;
	unsigned int    biCompression;
	unsigned int    biSizeImage;
	int             biXPelsPerMeter;
	int             biYPelsPerMeter;
	unsigned int    biClrUsed;
	unsigned int    biClrImportant;
} BITMAPHEADER;

typedef struct
{
	unsigned char rgbBlue;
	unsigned char rgbGreen;
	unsigned char rgbRed;
} RGBQUAD;

BITMAPHEADER read_headerBMP(FILE* f_input)
{	
	BITMAPHEADER HeaderOfImage;
	fread(&HeaderOfImage.bfType, 2, 1, f_input);
	fread(&HeaderOfImage.bfSize, 4, 1, f_input);
	fread(&HeaderOfImage.bfReserved1, 2, 1, f_input);
	fread(&HeaderOfImage.bfReserved2, 2, 1, f_input);
	fread(&HeaderOfImage.bfOffBits, 4, 1, f_input);

	fread(&HeaderOfImage.biSize, 4, 1, f_input);
	fread(&HeaderOfImage.biWidth, 4, 1, f_input);
	fread(&HeaderOfImage.biHeight, 4, 1, f_input);
	fread(&HeaderOfImage.biPlanes, 2, 1, f_input);
	fread(&HeaderOfImage.biBitCount, 2, 1, f_input);
	fread(&HeaderOfImage.biCompression, 4, 1, f_input);
	fread(&HeaderOfImage.biSizeImage, 4, 1, f_input);
	fread(&HeaderOfImage.biXPelsPerMeter, 4, 1, f_input);
	fread(&HeaderOfImage.biYPelsPerMeter, 4, 1, f_input);
	fread(&HeaderOfImage.biClrUsed, 4, 1, f_input);
	fread(&HeaderOfImage.biClrImportant, 4, 1, f_input);

	return HeaderOfImage;
}

void write_headerBMP(BITMAPHEADER HeaderOfImage, FILE* f_output)
{
	fwrite(&HeaderOfImage.bfType, 2, 1, f_output);
	fwrite(&HeaderOfImage.bfSize, 4, 1, f_output);
	fwrite(&HeaderOfImage.bfReserved1, 2, 1, f_output);
	fwrite(&HeaderOfImage.bfReserved2, 2, 1, f_output);
	fwrite(&HeaderOfImage.bfOffBits, 4, 1, f_output);

	fwrite(&HeaderOfImage.biSize, 4, 1, f_output);
	fwrite(&HeaderOfImage.biWidth, 4, 1, f_output);
	fwrite(&HeaderOfImage.biHeight, 4, 1, f_output);
	fwrite(&HeaderOfImage.biPlanes, 2, 1, f_output);
	fwrite(&HeaderOfImage.biBitCount, 2, 1, f_output);
	fwrite(&HeaderOfImage.biCompression, 4, 1, f_output);
	fwrite(&HeaderOfImage.biSizeImage, 4, 1, f_output);
	fwrite(&HeaderOfImage.biXPelsPerMeter, 4, 1, f_output);
	fwrite(&HeaderOfImage.biYPelsPerMeter, 4, 1, f_output);
	fwrite(&HeaderOfImage.biClrUsed, 4, 1, f_output);
	fwrite(&HeaderOfImage.biClrImportant, 4, 1, f_output);
}

float** RGBInFloat(RGBQUAD** rgb, int Width, int Height)
{
	float** rgbFloat = new float*[Height];
	for(int i = 0; i < Height; i++)
	{
		rgbFloat[i] = new float[Width];
	}

	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			float b = 255.0 / 2.0;
			float A = (rgb[i][j].rgbBlue + rgb[i][j].rgbGreen + rgb[i][j].rgbRed) / 3;
			if(A < b)  rgbFloat[i][j] = 1.0;
			else rgbFloat[i][j] = 0.0;
		}
	}

	return rgbFloat;
}

RGBQUAD** FloatInRGB(float** rgbFloat, int Width, int Height)
{
	RGBQUAD** rgb = new RGBQUAD*[Height];
	for(int i = 0; i < Height; i++)
	{
		rgb[i] = new RGBQUAD[Width];
	}

	for (int i = 0; i < Height; i++)
	{
		for (int j = 0; j < Width; j++)
		{
			if(rgbFloat[i][j] == 1.0)  rgb[i][j].rgbBlue = rgb[i][j].rgbGreen = rgb[i][j].rgbRed = 0;
			else rgb[i][j].rgbBlue = rgb[i][j].rgbGreen = rgb[i][j].rgbRed = 255;
		}
	}

	return rgb;
}
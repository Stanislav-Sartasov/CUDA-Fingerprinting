#include <iostream> 
#include <iomanip> 
#include <vector>
using namespace std;

typedef struct
{
	unsigned int    bfType;
	unsigned long   bfSize;
	unsigned int    bfReserved1;
	unsigned int    bfReserved2;
	unsigned long   bfOffBits;

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
	unsigned int rgbBlue;
	unsigned int rgbGreen;
	unsigned int rgbRed;
} RGBQUAD;

static unsigned short read_u16(FILE *fp)
{
	unsigned char b0, b1;

	b0 = getc(fp);
	b1 = getc(fp);

	return ((b1 << 8) | b0);
}

static unsigned int read_u32(FILE *fp)
{
	unsigned char b0, b1, b2, b3;

	b0 = getc(fp);
	b1 = getc(fp);
	b2 = getc(fp);
	b3 = getc(fp);

	return ((((((b3 << 8) | b2) << 8) | b1) << 8) | b0);
}

static int read_s32(FILE *fp)
{
	unsigned char b0, b1, b2, b3;

	b0 = getc(fp);
	b1 = getc(fp);
	b2 = getc(fp);
	b3 = getc(fp);

	return ((int)(((((b3 << 8) | b2) << 8) | b1) << 8) | b0);
}

std::vector< std::vector<RGBQUAD> > loadImage(const char* address)
{
	FILE * pFile = fopen(address, "rb");

	// считываем заголовок файла
	BITMAPHEADER header;

	header.bfType = read_u16(pFile);
	header.bfSize = read_u32(pFile);
	header.bfReserved1 = read_u16(pFile);
	header.bfReserved2 = read_u16(pFile);
	header.bfOffBits = read_u32(pFile);

	header.biSize = read_u32(pFile);
	header.biWidth = read_s32(pFile);
	header.biHeight = read_s32(pFile);
	header.biPlanes = read_u16(pFile);
	header.biBitCount = read_u16(pFile);
	header.biCompression = read_u32(pFile);
	header.biSizeImage = read_u32(pFile);
	header.biXPelsPerMeter = read_s32(pFile);
	header.biYPelsPerMeter = read_s32(pFile);
	header.biClrUsed = read_u32(pFile);
	header.biClrImportant = read_u32(pFile);

	std::vector< std::vector<RGBQUAD>> rgb(header.biWidth, std::vector<RGBQUAD>(header.biHeight)); // мы создали пустой вектор

	for (int i = 0; i < header.biWidth; i++) {
		for (int j = 0; j < header.biHeight; j++) {
			rgb[i][j].rgbBlue = getc(pFile);
			rgb[i][j].rgbGreen = getc(pFile);
			rgb[i][j].rgbRed = getc(pFile);
		}

		// пропускаем последний байт в строке
		getc(pFile);
	}
	printf("Width = %d, Height = %d\n", header.biWidth, header.biHeight);
	fclose(pFile);
	return rgb;
}

std::vector< std::vector<int>> ImageToVector(std::vector< std::vector<RGBQUAD> > Image)
{
	int Width = Image.size();
	int Height = Image[0].size();

	std::vector< std::vector<int>> rgb(Width, std::vector<int>(Height));

	for (int i = 0; i < Width; i++)
		for (int j = 0; j < Height; j++)
		{
			int color = (Image[i][j].rgbBlue + Image[i][j].rgbGreen + Image[i][j].rgbRed) / 3;
			if (color < 255 / 2)
				rgb[i][j] = 1;
			else rgb[i][j] = 0;
		}
	return rgb;
}

std::vector< std::vector<RGBQUAD> > VectorToImage(std::vector< std::vector<int>> rgb)
{
	int Width = rgb.size();
	int Height = rgb[0].size();
	std::vector< std::vector<RGBQUAD>> Image(Width, std::vector<RGBQUAD>(Height));

	for (int i = 0; i < Width; i++)
		for (int j = 0; j < Height; j++)
		{
			if (rgb[i][j] == 0)
				Image[i][j].rgbRed = Image[i][j].rgbGreen = Image[i][j].rgbBlue = 255;
			else Image[i][j].rgbRed = Image[i][j].rgbGreen = Image[i][j].rgbBlue = 0;
		}


	return Image;
}
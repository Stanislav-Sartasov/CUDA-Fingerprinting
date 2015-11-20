#include <iostream>
#include <xmmintrin.h>
#include "BMP.h"
#include "ModifiedJangSuen.h"
#include "OPATA.h"
#include <string>

int main()
{
	cout << "Enter  address of file:\n";
	char path[500];
	gets(path);
	FILE* f_input;

	f_input = fopen(path, "rb");

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

	std::vector< std::vector<RGBQUAD>> rgb(HeaderOfImage.biHeight, std::vector<RGBQUAD>(HeaderOfImage.biWidth)); // мы создали пустой вектор

	for (int i = 0; i < HeaderOfImage.biHeight; i++)
	{
		for (int j = 0; j < HeaderOfImage.biWidth; j++)
		{
			fread(&rgb[i][j].rgbBlue, 1, 1, f_input);
			fread(&rgb[i][j].rgbGreen, 1, 1, f_input);
			fread(&rgb[i][j].rgbRed, 1, 1, f_input);
		}
		fseek(f_input, HeaderOfImage.biWidth % 4, SEEK_CUR);
	}
	fclose(f_input);

	/*std::vector< std::vector<int>> matrix = ImageToVector(rgb);

	//cout << "Please, choose the algorythm:\n1.Modified JangSuen\n2.OPATA";

	//int chosen;
	//cin >> chosen;

	switch (chosen)
	{
	case 1:
	{
		//matrix = ModifiedJangSuen(matrix);
		break;
	}
	case 2:
	{
		break;
	}
	default:
		break;
	}

	//rgb = VectorToImage(matrix);

	cout << "Enter address of file:\n";
	*/
	FILE* f_output = fopen("NewPicture.bmp", "wb");
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

	for (int i = 0; i < HeaderOfImage.biHeight; i++)
	{
		for (int j = 0; j < HeaderOfImage.biWidth; j++)
		{
			fwrite(&rgb[i][j].rgbBlue, 1, 1, f_output);
			fwrite(&rgb[i][j].rgbGreen, 1, 1, f_output);
			fwrite(&rgb[i][j].rgbRed, 1, 1, f_output);
		}
		fwrite("0", HeaderOfImage.biWidth % 4, 1, f_output);
	}

	fclose(f_output);

	cout << "Success!!!\n";

	system("pause");
	return 0;
}
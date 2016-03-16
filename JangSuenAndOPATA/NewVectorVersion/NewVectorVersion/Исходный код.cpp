#include <iostream>
#include <xmmintrin.h>
#include "BMP.h"
#include "ModifiedJangSuen.h"
#include "OPATA.h"
#include <string>
#pragma warning(disable : 4996)
 
int main()
{
	unsigned char a = '8';
	unsigned char b = '9';
	float B = (float)a;
	//cout << B << "\n";
	a = a +'9';
	a++;
	a++;
	B = (float)a;
	//cout << a + b + "\n";
	
	int S = 8;
	a = '10';
	B = (float)S;
	cout << B << "\n";

	a = '11';
	B = (float)a;
	cout << B << "\n";

	a = '12';
	B = static_cast<float> ( a);

	cout << B << "\n";

	a = '13';
	B = (float)a;
	cout << B << "\n";
	
	cout << "Enter  address of file:\n";
	char path[500];
	gets_s(path);
	FILE* f_input = fopen(path, "r+b");

	BITMAPHEADER HeaderOfImage = read_headerBMP(f_input);
	unsigned char R;
	RGBQUAD** rgb = new RGBQUAD*[HeaderOfImage.biHeight];
	for (int j = 0; j < HeaderOfImage.biHeight; j++)
		rgb[j] = new RGBQUAD[HeaderOfImage.biWidth];
	for (int i = 0; i < HeaderOfImage.biHeight; i++)
	{
		for (int j = 0; j < HeaderOfImage.biWidth; j++)
		{
			fread(&rgb[i][j].rgbBlue, 1, 1, f_input);
			fread(&rgb[i][j].rgbGreen, 1, 1, f_input);
			fread(&rgb[i][j].rgbRed, 1, 1, f_input);
			fread(&R, 1, 1, f_input);
		}
	}	
	fclose(f_input);
	
	float** pic = RGBInFloat(rgb, HeaderOfImage.biWidth, HeaderOfImage.biHeight);
	pic = ModifiedJangSuen(pic);
	//cout << "Enter address of file:\n";

	rgb = FloatInRGB(pic, HeaderOfImage.biWidth, HeaderOfImage.biHeight);

	FILE* f_output = fopen("NewPicture.bmp", "w+b");
	write_headerBMP(HeaderOfImage, f_output);
	for (int i = 0; i < HeaderOfImage.biHeight; i++)
	{
		for (int j = 0; j < HeaderOfImage.biWidth; j++)
		{
			fwrite(&rgb[i][j].rgbBlue, 1, 1, f_output);
			fwrite(&rgb[i][j].rgbGreen, 1, 1, f_output);
			fwrite(&rgb[i][j].rgbRed, 1, 1, f_output);
			fwrite("0", 1, 1, f_output);
		}
	}
	fclose(f_output);

	cout << "Picture Is Ready!!!\n";
	system("pause");
	return 0;
}
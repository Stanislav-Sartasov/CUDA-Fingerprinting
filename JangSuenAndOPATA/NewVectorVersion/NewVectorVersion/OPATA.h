#include <iostream>
#include <xmmintrin.h>
#include "defines.h"

float* t = new float[4];
float* o = new float[4];
float* z = new float[4];
__m128 Two;
__m128 One;
__m128 Zero;

float* patA = new float[4];
float* patB = new float[4];
float* patC = new float[4];
float* patD = new float[4];
float* patE = new float[4];
float* patF = new float[4];
float* patG = new float[4];
float* patH = new float[4];
float* patI = new float[4];
float* patJ = new float[4];
float* patK = new float[4];
float* patL = new float[4];
float* patM = new float[4];
float* patN = new float[4];

__m128 PatternA;
__m128 PatternB;
__m128 PatternC;
__m128 PatternD;
__m128 PatternE;
__m128 PatternF;
__m128 PatternG;
__m128 PatternH;
__m128 PatternI;
__m128 PatternJ;
__m128 PatternK;
__m128 PatternL;
__m128 PatternM;
__m128 PatternN;

void Initialization()
{
	for(int i = 0; i < 4; i++)
	{
		t[i] = 2.0;
		o[i] = 1.0;
		z[i] = 0.0;

		patA[i] = 11.0;
		patB[i] = 12.0;
		patC[i] = 13.0;
		patD[i] = 14.0;
		patE[i] = 15.0;
		patF[i] = 16.0;
		patG[i] = 17.0;
		patH[i] = 18.0;
		patI[i] = 19.0;
		patJ[i] = 20.0;
		patK[i] = 21.0;
		patL[i] = 22.0;
		patM[i] = 23.0;
		patN[i] = 24.0;
	}
	float t1[4] = {2.0,2.0,2.0,2.0};
	Two = _mm_load_ps(t);
	One = _mm_load_ps(o);
	Zero = _mm_load_ps(z);

	PatternA = _mm_load_ps(patA);
	PatternB = _mm_load_ps(patB);
	PatternC = _mm_load_ps(patC);
	PatternD = _mm_load_ps(patD);
	PatternE = _mm_load_ps(patE);
	PatternF = _mm_load_ps(patF);
	PatternG = _mm_load_ps(patG);
	PatternH = _mm_load_ps(patH);
	PatternI = _mm_load_ps(patI);
	PatternJ = _mm_load_ps(patJ);
	PatternK = _mm_load_ps(patK);
	PatternL = _mm_load_ps(patL);
	PatternM = _mm_load_ps(patM);
	PatternN = _mm_load_ps(patN);
}

struct Move
{
	int XOffset;
	int YOffset;

	Move()
	{}

	Move(int x, int y)
	{
		XOffset = x;
		YOffset = y;
	}
};

__m128 IsComparedO(__m128 Neighborhood[4][4])
{
	__m128 comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	__m128 comparing04 = _mm_cmpeq_ps(Neighborhood[1][0], One);
	__m128 comparing10 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	__m128 comparing12 = _mm_cmpeq_ps(Neighborhood[2][0], One);
	__m128 comparing13 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	__m128 comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	__m128 comparing23 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	__m128 comparing30 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	__m128 comparing31 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	__m128 comparing32 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	__m128 comparing33 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);

	__m128 Compare = _mm_and_ps(comparing01, comparing04);
	Compare = _mm_and_ps(Compare, comparing10);
	Compare = _mm_and_ps(Compare, comparing12);
	Compare = _mm_and_ps(Compare, comparing13);
	Compare = _mm_and_ps(Compare, comparing21);
	Compare = _mm_and_ps(Compare, comparing23);
	Compare = _mm_and_ps(Compare, comparing30);
	Compare = _mm_and_ps(Compare, comparing31);
	Compare = _mm_and_ps(Compare, comparing32);
	Compare = _mm_and_ps(Compare, comparing33);

	return Compare;
}

__m128** IsConcaveCorner(__m128 A[3][3], __m128 Pattern)
{
	__m128 C[3][3];
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			C[i][j] = Zero;
		}

		__m128 ForPatternA = _mm_cmpeq_ps(Pattern, PatternA);
		__m128 ForA1 = _mm_cmpeq_ps(A[0][2], One);
		__m128 ForA2 = _mm_cmpeq_ps(A[2][2], One);
		C[0][1] = _mm_and_ps(ForPatternA, ForA1);
		C[2][1] = _mm_and_ps(ForPatternA, ForA2);

		__m128 ForPatternB = _mm_cmpeq_ps(Pattern, PatternB);
		ForA1 = _mm_cmpeq_ps(A[2][2], One);
		ForA2 = _mm_cmpeq_ps(A[2][0], One);
		C[1][2] = _mm_and_ps(ForPatternB, ForA1);
		C[1][0] = _mm_and_ps(ForPatternB, ForA2);

		__m128 ForPatternC = _mm_cmpeq_ps(Pattern, PatternC);
		ForA1 = _mm_cmpeq_ps(A[2][0], One);
		ForA2 = _mm_cmpeq_ps(A[0][0], One);
		C[2][1] = _mm_and_ps(ForPatternC, ForA1);
		C[1][2] = _mm_and_ps(ForPatternC, ForA2);

		__m128 ForPatternD = _mm_cmpeq_ps(Pattern, PatternD);
		ForA1 = _mm_cmpeq_ps(A[0][2], One);
		ForA2 = _mm_cmpeq_ps(A[0][0], One);
		C[1][2] = _mm_and_ps(ForPatternD, ForA1);
		C[1][0] = _mm_and_ps(ForPatternD, ForA2);

		__m128 ForPatternE = _mm_cmpeq_ps(Pattern, PatternE);
		ForA1 = _mm_and_ps(_mm_cmpeq_ps(A[0][0], One), _mm_cmpeq_ps(A[2][0], One));
		ForA2 = _mm_and_ps(_mm_cmpeq_ps(A[2][0], One), _mm_cmpeq_ps(A[2][2], One));
		C[1][0] = _mm_and_ps(ForPatternE, ForA1);
		C[2][1] = _mm_and_ps(ForPatternE, ForA2);

		__m128 ForPatternF = _mm_cmpeq_ps(Pattern, PatternF);
		ForA1 = _mm_cmpeq_ps(A[0][2], One);
		ForA2 = _mm_cmpeq_ps(A[2][0], One);
		C[0][1] = _mm_and_ps(ForPatternF, ForA1);
		C[1][2] = _mm_and_ps(ForPatternF, ForA2);

		__m128 ForPatternH = _mm_cmpeq_ps(Pattern, PatternH);
		ForA1 = _mm_and_ps(_mm_cmpeq_ps(A[0][0], One), _mm_cmpeq_ps(A[0][2], One));
		ForA2 = _mm_and_ps(_mm_cmpeq_ps(A[0][0], One), _mm_cmpeq_ps(A[2][0], One));
		C[0][1] = _mm_and_ps(ForPatternH, ForA1);
		C[1][0] = _mm_and_ps(ForPatternH, ForA2);

		__m128 ForPatternI = _mm_cmpeq_ps(Pattern, PatternI);
		ForA1 = _mm_cmpeq_ps(A[0][2], One);
		ForA2 = _mm_cmpeq_ps(A[2][0], One);
		C[1][2] = _mm_and_ps(ForPatternI, ForA1);
		C[2][1] = _mm_and_ps(ForPatternI, ForA2);

		__m128** C1 = (__m128**) C;
		return C1;
}

__m128 IsComparedPatterns(__m128 Neighborhood[3][3], __m128 p8, __m128 p9)
{
	__m128 comparing = Zero;

	__m128 comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], One);
	__m128 comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	__m128 comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], One);
	__m128 comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], Zero);
	__m128 comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], One);
	__m128 comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	__m128 comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	__m128 comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	__m128 Y = _mm_or_ps(comparing02, comparing22);
	__m128 CompareA = _mm_and_ps(Y, _mm_and_ps(comparing12, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing00, comparing01))))));
	CompareA = _mm_and_ps(PatternA, CompareA);

	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], One);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], One);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	__m128 CompareB = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing00, comparing01)))))));
	CompareB = _mm_and_ps(PatternB, CompareB);
	comparing = _mm_add_ps(CompareA, CompareB);

	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	__m128 comparingP8 = _mm_cmpeq_ps(p8, One);
	Y = _mm_or_ps(comparing00, comparing20);
	__m128 CompareC = _mm_and_ps(comparingP8, _mm_and_ps(Y, _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, comparing01)))))));
	CompareC = _mm_and_ps(PatternC, CompareC);
	comparing = _mm_add_ps(comparing, CompareC);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	__m128 comparingP9 = _mm_cmpeq_ps(p9, One);
	Y = _mm_or_ps(comparing00, comparing02);
	__m128 CompareD = _mm_and_ps(comparingP9, _mm_and_ps(Y, _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, comparing01)))))));
	CompareD = _mm_and_ps(PatternD, CompareD);
	comparing = _mm_add_ps(comparing, CompareD);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], One);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], Zero);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	__m128 X = _mm_xor_ps(comparing22, _mm_xor_ps(comparing00, comparing20));
	__m128 CompareE = _mm_and_ps(X, _mm_and_ps(comparing10, _mm_and_ps(comparing21, _mm_and_ps(comparing12, _mm_and_ps(comparing02, comparing01)))));
	CompareE = _mm_and_ps(PatternE, CompareE);
	comparing = _mm_add_ps(comparing, CompareE);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	X = _mm_xor_ps(comparing22, comparing00);
	__m128 CompareF = _mm_and_ps(X, _mm_and_ps(comparing10, _mm_and_ps(comparing21, _mm_and_ps(comparing12, _mm_and_ps(comparing02, comparing01)))));
	CompareF = _mm_and_ps(PatternF, CompareF);
	comparing = _mm_add_ps(comparing, CompareF);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	__m128 CompareG = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing01, comparing00)))))));
	CompareG = _mm_and_ps(PatternG, CompareG);
	comparing = _mm_add_ps(comparing, CompareG);

	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], One);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], Zero);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	X = _mm_xor_ps(comparing20, _mm_xor_ps(comparing00, comparing02));
	__m128 CompareH = _mm_and_ps(X, _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing12, _mm_and_ps(comparing10, comparing01)))));
	CompareH = _mm_and_ps(PatternH, CompareH);
	comparing = _mm_add_ps(comparing, CompareH);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	X = _mm_xor_ps(comparing02, comparing20);
	__m128 CompareI = _mm_and_ps(X, _mm_and_ps(comparing22, (_mm_and_ps(comparing21, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing01, comparing00)))))));
	CompareI = _mm_and_ps(PatternI, CompareI);
	comparing = _mm_add_ps(comparing, CompareI);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	__m128 CompareJ = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing01, comparing00)))))));
	CompareJ = _mm_and_ps(PatternJ, CompareJ);
	comparing = _mm_add_ps(comparing, CompareJ);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], Zero);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], One);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], One);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	__m128 CompareK = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing01, comparing00)))))));
	CompareK = _mm_and_ps(PatternK, CompareK);
	comparing = _mm_add_ps(comparing, CompareK);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], One);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], Zero);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], One);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], Zero);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], One);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	__m128 CompareL = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing01, comparing00)))))));
	CompareL = _mm_and_ps(PatternL, CompareL);
	comparing = _mm_add_ps(comparing, CompareL);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], One);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], One);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], Zero);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], Zero);
	__m128 CompareM = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing01, comparing00)))))));
	CompareM = _mm_and_ps(PatternM, CompareM);
	comparing = _mm_add_ps(comparing, CompareM);


	comparing00 = _mm_cmpeq_ps(Neighborhood[0][0], Zero);
	comparing01 = _mm_cmpeq_ps(Neighborhood[0][1], Zero);
	comparing02 = _mm_cmpeq_ps(Neighborhood[0][2], One);
	comparing10 = _mm_cmpeq_ps(Neighborhood[1][0], Zero);
	comparing12 = _mm_cmpeq_ps(Neighborhood[1][2], One);
	comparing20 = _mm_cmpeq_ps(Neighborhood[2][0], Zero);
	comparing21 = _mm_cmpeq_ps(Neighborhood[2][1], Zero);
	comparing22 = _mm_cmpeq_ps(Neighborhood[2][2], One);
	__m128 CompareN = _mm_and_ps(comparing22, _mm_and_ps(comparing21, _mm_and_ps(comparing20, _mm_and_ps(comparing12, _mm_and_ps(comparing10, _mm_and_ps(comparing02, _mm_and_ps(comparing01, comparing00)))))));
	CompareN = _mm_and_ps(PatternN, CompareN);
	comparing = _mm_add_ps(comparing, CompareN);

	return comparing;
}

float** OPATA(float** Image)
{
	Initialization();

	Move* offsets = new Move[4];

	offsets[0] = Move(-1, 0);
	offsets[1] = Move(1, 0);
	offsets[2] = Move(0, -1);
	offsets[3] = Move(0, 1);

	int k = 0;
	int number = 1;
	int Heigth = sizeof(Image[0]) / 4;
	int Width = (sizeof(Image) / 4) / Heigth;
	__m128 flag = One;

	__m128** ImageV = (__m128**)Image;
	__m128** Mark = new __m128*[Width / 4];
	for (int k = 0; k < Width / 4; k++)
	{
		Mark[k] = new __m128[Heigth / 4];
	}

	for (int k = 0; k < Width / 4; k++)
		for (int l = 0; l < Heigth / 4; l++)
		{
			Mark[k][l] = Zero;
		}

		int i = 0;

		do
		{
			for (int i = 1; i < Width / 4; i++)
				for (int j = 1; j < Heigth / 4; j++)
				{
					__m128 neighborhood[3][3] = { { ImageV[i - 1][j - 1], ImageV[i - 1][j], ImageV[i - 1][j + 1] },
					{ ImageV[i][j - 1], ImageV[i][j], ImageV[i][j + 1] },
					{ ImageV[i + 1][j - 1], ImageV[i + 1][j], ImageV[i + 1][j + 1] }, };

					__m128 patterns = IsComparedPatterns(neighborhood, ImageV[i][j + 2], ImageV[i + 2][i]);

					ImageV[i][j] = _mm_andnot_ps(patterns, ImageV[i][j]);

					for (int u = 0; u < 4; u++)
					{
						int col = i + offsets[u].XOffset;
						int row = j + offsets[u].YOffset;
						int x = 1 + offsets[u].XOffset;
						int y = 1 + offsets[u].YOffset;

						__m128** IsConcave = IsConcaveCorner(neighborhood, patterns);

						__m128 NeibourhoodForO[4][4] = { { ImageV[col - 2][row - 2], ImageV[col - 2][row - 1], ImageV[col - 2][row], ImageV[col - 2][row + 1] },
						{ ImageV[col - 1][row - 2], ImageV[col - 1][row - 1], ImageV[col - 1][row], ImageV[col - 1][row + 1] },
						{ ImageV[col][row - 2], ImageV[col][row - 1], ImageV[col][row], ImageV[col][row + 1] },
						{ ImageV[col + 1][row - 2], ImageV[col + 1][row - 1], ImageV[col + 1][row], ImageV[col + 1][row + 1] }, };


						__m128 cond = _mm_andnot_ps(IsComparedO(NeibourhoodForO), Mark[col][row]);
						ImageV[col][row] = _mm_and_ps(cond, IsConcave[x][y]);
					}
				}
		} while (_mm_ucomieq_ss(flag, Zero));
		return Image;
}
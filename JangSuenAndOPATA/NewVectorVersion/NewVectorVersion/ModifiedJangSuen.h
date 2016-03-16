#include <iostream>
#include <xmmintrin.h>
#include "defines.h"

float* t1 = new float[4];
float* o1 = new float[4];
float* z1 = new float[4];
float* s1 = new float[4];
__m128 Two1;
__m128 One1;
__m128 Zero1;


void Initialization1()
{
	for(int i = 0; i < 4; i++)
	{
		t1[i] = 2.0;
		o1[i] = 1.0;
		z1[i] = 0.0;
		s1[i] = 6.0;
	}
	Two1 = _mm_load_ps(t1);
	One1 = _mm_load_ps(o1);
	Zero1 = _mm_load_ps(z1);
}

__m128 countA1(__m128 Neighborhood[3][3])
{
	__m128 count = Zero1;
	for (int i = 1; i < 3; i++)
	{
		count = _mm_add_ps(count, _mm_and_ps(_mm_cmpeq_ps(Neighborhood[0][i - 1], Zero1), _mm_cmpeq_ps(Neighborhood[0][i],One1)));
	}

	for (int i = 1; i < 3; i++)
	{
		count = _mm_add_ps(count, _mm_and_ps(_mm_cmpeq_ps(Neighborhood[i - 1][2], Zero1), _mm_cmpeq_ps(Neighborhood[i][2], One1)));
	}

	for (int i = 2; i <= 1; i--)
	{
		count = _mm_add_ps(count, _mm_and_ps(_mm_cmpeq_ps(Neighborhood[2][i - 1], Zero1), _mm_cmpeq_ps(Neighborhood[2][i], One1)));
	}

	for (int i = 2; i <= 1; i--)
	{
		count = _mm_add_ps(count, _mm_and_ps(_mm_cmpeq_ps(Neighborhood[i - 1][0], Zero1), _mm_cmpeq_ps(Neighborhood[i][0], One1)));
	}

	return count;
}

__m128 countB1(__m128 Neighborhood[3][3])
{
	__m128 count = _mm_add_ps(Neighborhood[0][0], Neighborhood[0][1]);
	count = _mm_add_ps(count, Neighborhood[0][2]);

	for (int i = 1; i < 3; i++)
		for (int j = 0; j < 3; j++)
		{
			count = _mm_add_ps(count, Neighborhood[i][j]);
		}
	return count;
}

float** ModifiedJangSuen(float** Image)
{
	Initialization1();

	int k = 0;
	int number = 1;
	int Heigth = sizeof(Image[0]) / 4;
	int Width = (sizeof(Image) / 4) / Heigth;
	

	__m128** ImageV = (__m128**)Image;

	for (int i = 1; i < Width / 4; i++)
		for (int j = 1; j < Heigth / 4; j++)
		{
			__m128 neighborhood[3][3] = { { ImageV[i - 1][j - 1], ImageV[i - 1][j], ImageV[i - 1][j + 1] },
			{ ImageV[i][j - 1], ImageV[i][j], ImageV[i][j + 1] },
			{ ImageV[i + 1][j - 1], ImageV[i + 1][j], ImageV[i + 1][j + 1] }, };

			__m128 B1 = countB1(neighborhood);
			__m128 B1IN = _mm_and_ps(_mm_cmpge_ps(B1, Two1),
				_mm_cmpge_ps(_mm_load_ps(s1), B1));

			__m128 A1 = countA1(neighborhood);
			__m128 IsA1One = _mm_cmpeq_ps(A1, One1);
			__m128 IsA1Two = _mm_cmpeq_ps(A1, Two1);
			__m128 vicinity1, vicinity2;
			
			if ((number & 2) == 1)
			{
				vicinity1 = _mm_or_ps(_mm_mul_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i - 1][j + 1]), ImageV[i + 1][j + 1]),
					_mm_mul_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i - 1][j + 1]), ImageV[i + 1][j - 1]));

				__m128 condA1 = _mm_cmpeq_ps(_mm_mul_ps(ImageV[i - 1][j + 1], ImageV[i + 1][j + 1]), One1);
				condA1 = _mm_and_ps(condA1, _mm_cmpeq_ps(ImageV[i][j - 1], Zero1));

				__m128 condB1A = _mm_cmpeq_ps(_mm_mul_ps(ImageV[i - 1][j + 1], ImageV[i - 1][j - 1]), One1);
				__m128 condB1B = _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(One1, ImageV[i - 1][j]), _mm_sub_ps(One1, ImageV[i + 1][j - 1])), _mm_sub_ps(One1, ImageV[i + 1][j]));
				condB1B = _mm_cmpeq_ps(condB1A, One1);
				__m128 condB1 = _mm_and_ps(condB1A, condB1B);

				vicinity2 = _mm_or_ps(_mm_mul_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i - 1][j + 1]), ImageV[i + 1][j + 1]),
					_mm_mul_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i - 1][j + 1]), ImageV[i + 1][j - 1]));
			}
			else
			{
				vicinity1 = _mm_or_ps(_mm_mul_ps(_mm_mul_ps(ImageV[i - 1][j + 1], ImageV[i + 1][j + 1]), ImageV[i + 1][j - 1]),
					_mm_mul_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i + 1][j + 1]), ImageV[i + 1][j - 1]));

				__m128 condA2 = _mm_cmpeq_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i + 1][j - 1]), One1);
				condA2 = _mm_and_ps(condA2, _mm_cmpeq_ps(ImageV[i][j + 1], Zero1));

				__m128 condB2A = _mm_cmpeq_ps(_mm_mul_ps(ImageV[i - 1][j - 1], ImageV[i + 1][j - 1]), One1);
				__m128 condB2B = _mm_mul_ps(_mm_mul_ps(_mm_sub_ps(One1, ImageV[i - 1][j]), _mm_sub_ps(One1, ImageV[i - 1][j + 1])), _mm_sub_ps(One1, ImageV[i + 1][j]));
				condB2B = _mm_cmpeq_ps(condB2A, One1);
				__m128 condB2 = _mm_and_ps(condB2A, condB2B);

				vicinity2 = _mm_or_ps(condA2, condB2);
			}

			__m128 Conditional1 = _mm_and_ps(_mm_and_ps(B1IN, vicinity1), IsA1One);
			__m128 Conditional2 = _mm_and_ps(_mm_and_ps(B1IN, vicinity2), IsA1Two);
			__m128 Conditional = _mm_xor_ps(Conditional1, Conditional2);
			ImageV[i][j] = _mm_andnot_ps(Conditional, ImageV[i][j]);

			number++;
		}

	return Image;
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace CUDAFingerprinting.OrientationField
{
    public class Block16
    {
        public const int SIZE = 16;
        public double[,] pxl;			// пиксели блока
		public double orientation;		// массив направлений
        double[,] Gx;					// градиенты
        double[,] Gy;

        public Block16()
        {
            this.pxl = new double[SIZE, SIZE];
        }

        public void SetOrientation()
        {
            double[,] filterX = new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            double[,] filterY = new double[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

            Gx =  CUDAFingerprinting.Common.ConvolutionHelper.Convolve(pxl, filterX); //GradientX(pxl);
			Gy =  CUDAFingerprinting.Common.ConvolutionHelper.Convolve(pxl, filterY); //GradientY(pxl)
            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < SIZE; i++)
            {
                for (int j = 0; j < SIZE; j++)
                {
                    numerator += Gx[i, j] * Gy[i, j];
                    denominator += Gx[i, j] * Gx[i, j] - Gy[i, j] * Gy[i, j];
                }
            }
            if (denominator == 0)
            {
                this.orientation = Math.PI / 2;
            }
            else
            {
            this.orientation = Math.Atan2(2 * numerator, denominator) / 2;
            }
        }

		//public double[,] GradientX(double[,] bytes)           // возвращает массив градиентов по X после примененения маски Собеля
		//{
		//	int[,] filter = new int[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
		//	double[,] result = new double[bytes.GetUpperBound(0) + 1, bytes.GetUpperBound(1) + 1];

		//	for (int x = 0; x < bytes.GetUpperBound(0) + 1; x++)
		//	{
		//		for (int y = 0; y < bytes.GetUpperBound(1) + 1; y++)
		//		{
		//			for (int i = -1; i < 2; i++)
		//			{
		//				for (int j = -1; j < 2; j++)
		//				{
		//					// если не выходим за границы
		//					if (x + i >= 0 && x + i <= bytes.GetUpperBound(0) &&
		//						y + j >= 0 && y + j <= bytes.GetUpperBound(1))
		//					{
		//						result[x, y] += filter[i + 1, j + 1] * bytes[x + i, y + j];
		//					}
		//				}
		//			}
		//		}
		//	}
		//	return result;
		//}

		//public double[,] GradientY(double[,] bytes)           // возвращает массив градиентов по Y после примененения маски Собеля
		//{
		//	int[,] filter = new int[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
		//	double[,] result = new double[bytes.GetUpperBound(0) + 1, bytes.GetUpperBound(1) + 1];

		//	for (int x = 0; x < bytes.GetUpperBound(0) + 1; x++)
		//	{
		//		for (int y = 0; y < bytes.GetUpperBound(1) + 1; y++)
		//		{
		//			for (int i = -1; i < 2; i++)
		//			{
		//				for (int j = -1; j < 2; j++)
		//				{
		//					// если не выходим за границы
		//					if (x + i >= 0 && x + i <= bytes.GetUpperBound(0) &&
		//						y + j >= 0 && y + j <= bytes.GetUpperBound(1))
		//					{
		//						result[x, y] += filter[i + 1, j + 1] * bytes[x + i, y + j];
		//					}
		//				}
		//			}
		//		}
		//	}
		//	return result;
		//}
    }



    public class OrientationField
    {
        public Block16[,] blocks;
        public const int SIZE = 16;

        public OrientationField(byte[,] bytes)
        {
            int maxX = bytes.GetUpperBound(0) + 1;
            int maxY = bytes.GetUpperBound(1) + 1;
            // разделение на блоки
            this.blocks = new Block16[(int)Math.Ceiling((float)(maxX / SIZE)), (int)Math.Ceiling((float)(maxY / SIZE))];
            for (int i = 0; i < blocks.GetUpperBound(0) + 1; i++)
            {
                for (int j = 0; j < blocks.GetUpperBound(1) + 1; j++)
                {
                    blocks[i, j] = new Block16();
                    // копируем в блоки части массива
					for (int x = 0; x < SIZE; x++)
					{
						for (int y = 0; y < SIZE; y++)
						{
							blocks[i, j].pxl[x, y] = bytes[i * SIZE + x, j * SIZE + y];
						}
					}
					// вычисляем направления
                    blocks[i, j].SetOrientation();
                }
            }
        }

		public double GetOrientation(int x, int y)                  // метод, определяющий по входным координатам (х, у) поле напрваления в этой точке
		{
			int row = x / SIZE;
			int column = y / SIZE;
			return this.blocks[row, column].orientation;
		}
    } 
}

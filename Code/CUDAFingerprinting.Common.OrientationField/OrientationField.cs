using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace CUDAFingerprinting.Common.OrientationField
{
    public class Block
    {
        private int _size;
        private double[,] _pxl;			// пиксели блока
		private double _orientation;		// массив направлений

		public double[,] Pxl
		{
			get { return _pxl; }
		}
		public double Orientation
		{
			get { return _orientation; }
		}
		public int Size
		{
			get { return _size; }
		}

        public Block(int SIZE, int[,] bytes, int i, int j)		// параметры: размер блока; массив пикселей, из которого копировать в блок; координаты, с которых начать копирование
        {
			this._size = SIZE;
            this._pxl = new double[SIZE, SIZE];
			// копируем в блоки части массива
			for (int x = 0; x < SIZE; x++)
			{
				for (int y = 0; y < SIZE; y++)
				{
					this._pxl[x, y] = bytes[x + i, y + j];
				}
			}
			// вычисляем направления
			this.SetOrientation();
        }

        public void SetOrientation()
        {
            double[,] filterX = new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            double[,] filterY = new double[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };

			// градиенты
			double[,] Gx = CUDAFingerprinting.Common.ConvolutionHelper.Convolve(_pxl, filterX); 
			double[,] Gy = CUDAFingerprinting.Common.ConvolutionHelper.Convolve(_pxl, filterY); 
            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < _size; i++)
            {
                for (int j = 0; j < _size; j++)
                {
                    numerator += Gx[i, j] * Gy[i, j];
                    denominator += Gx[i, j] * Gx[i, j] - Gy[i, j] * Gy[i, j];
                }
            }
            if (denominator == 0)
            {
                this._orientation = Math.PI / 2;
            }
            else
            {
            this._orientation = Math.Atan2(2 * numerator, denominator) / 2;
            }
        }
    }

    public class OrientationField
    {
        Block[,] _blocks;
        const int _SIZE = 16;

		// property
		public Block[,] Blocks
		{
			get
			{
				return _blocks;
			}
		}
		public int SIZE
		{
			get
			{
				return _SIZE;
			}
		}

        public OrientationField(int[,] bytes)
        {
            int maxX = bytes.GetUpperBound(0) + 1;
            int maxY = bytes.GetUpperBound(1) + 1;
            // разделение на блоки: количество строк и колонок
            this._blocks = new Block[(int)Math.Floor((float)(maxY / _SIZE)), (int)Math.Floor((float)(maxX / _SIZE))];
            for (int row = 0; row < _blocks.GetUpperBound(0) + 1; row++)
            {
                for (int column = 0; column < _blocks.GetUpperBound(1) + 1; column++)
                {
					_blocks[row, column] = new Block(_SIZE, bytes, column * _SIZE, row * _SIZE);
                }
            }

        }

		public double GetOrientation(int x, int y)                  // метод, определяющий по входным координатам (х, у) поле напрваления в этой точке
		{
			int row = y / _SIZE;
			int column = x / _SIZE;
			return this._blocks[row, column].Orientation;
		}
    } 
}

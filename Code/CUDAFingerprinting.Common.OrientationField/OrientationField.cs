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

        public Block(int size, int[,] bytes, int i, int j)		// параметры: размер блока; массив пикселей, из которого копировать в блок; координаты, с которых начать копирование
        {
            this._size = size;
            this._pxl = new double[size, size];
			// копируем в блоки части массива
            for (int x = 0; x < size; x++)
			{
                for (int y = 0; y < size; y++)
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
			double[,] Gx = ConvolutionHelper.Convolve(_pxl, filterX); 
			double[,] Gy = ConvolutionHelper.Convolve(_pxl, filterY); 
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
        public const int DefaultSize = 16;

		// property
		public Block[,] Blocks
		{
			get
			{
				return _blocks;
			}
		}
        public int BlockSize
        {
            get;
            private set;
        }

        public OrientationField(int[,] bytes, int blockSize)
        {
            BlockSize = blockSize;
            int maxX = bytes.GetUpperBound(0) + 1;
            int maxY = bytes.GetUpperBound(1) + 1;
            // разделение на блоки: количество строк и колонок
            this._blocks = new Block[(int)Math.Floor((float)(maxY / BlockSize)), (int)Math.Floor((float)(maxX / BlockSize))];
            for (int row = 0; row < _blocks.GetUpperBound(0) + 1; row++)
            {
                for (int column = 0; column < _blocks.GetUpperBound(1) + 1; column++)
                {
                    _blocks[row, column] = new Block(BlockSize, bytes, column * BlockSize, row * BlockSize);
                }
            }
        }

        public OrientationField(int[,] bytes):this(bytes, DefaultSize)
        {
        }

		public double GetOrientation(int x, int y)                  // метод, определяющий по входным координатам (х, у) поле напрваления в этой точке
		{
            int row = y / BlockSize;
            int column = x / BlockSize;
			return this._blocks[row, column].Orientation;
		}
    } 
}

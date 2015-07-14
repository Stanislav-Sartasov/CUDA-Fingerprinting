using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;


namespace CUDAFingerprinting.Common
{
    public class Block
    {
        private int _size;
        private double[,] _gx;			// пиксели градиентов блока
		private double[,] _gy;		
		private double _orientation;		// направление в данном блоке

		public double[,] Gx
		{
			get { return _gx; }
		}
		public double[,] Gy
		{
			get { return _gy; }
		}
		public double Orientation
		{
			get { return _orientation; }
            set { _orientation = value; }
		}
		public int Size
		{
			get { return _size; }
		}

        public Block(int size, double[,] gradientX, double[,] gradientY, int i, int j)		// параметры: размер блока; градиенты; координаты, с которых начать копирование
        {
            this._size = size;
            this._gx = new double[size, size];
			this._gy = new double[size, size];
			// копируем необходимые части градиентов
            for (int x = 0; x < size; x++)
			{
                for (int y = 0; y < size; y++)
				{
					this._gx[x, y] = gradientX[x + i, y + j];
					this._gy[x, y] = gradientY[x + i, y + j];
				}
			}
			// вычисляем направления
			_orientation = this.SetOrientation();
        }

		public Block(double[,] Orientation, int size, double[,] gradientX, double[,] gradientY, int centerRow, int centerColumn)				// вычисляет направление в пикселе [i, j], результат помещает в Orientation
		{
			this._size = size;
			this._gx = new double[size, size];
			this._gy = new double[size, size];
			int center = size / 2;
			int upperLimit = center - 1;
			// копируем необходимые части градиентов
			for (int i = -center; i <= upperLimit; i++)
			{
				for (int j = -center; j <= upperLimit; j++)
				{
					if (i + centerRow < 0 || i + centerRow >= gradientX.GetUpperBound(0) || j + centerColumn < 0 || j + centerColumn >= gradientX.GetUpperBound(1))
					{	// выход за пределы картинки
						_gx[i + center, j + center] = 0;
						_gy[i + center, j + center] = 0;
					}
					else
					{
						_gx[i + center, j + center] = gradientX[i + centerRow, j + centerColumn];
						_gy[i + center, j + center] = gradientY[i + centerRow, j + centerColumn];
					}
				}
			}

			Orientation[centerRow, centerColumn] = _orientation = SetOrientation();
		}

        public double SetOrientation()
        {
            double numerator = 0;
            double denominator = 0;
            for (int i = 0; i < _size; i++)
            {
                for (int j = 0; j < _size; j++)
                {
                    numerator += _gx[i, j] * _gy[i, j];
					denominator += _gx[i, j] * _gx[i, j] - _gy[i, j] * _gy[i, j];
                }
            }
            if (denominator == 0)
            {
                return Math.PI / 2;
            }
            else
            {
                double orientation = Math.PI / 2 + Math.Atan2(2 * numerator, denominator) / 2;
				if (orientation > Math.PI / 2) orientation -= Math.PI;
				return orientation;
            }
        }
	}

	// ------------------------------------------------------------------------------------

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

		public OrientationField(int[,] bytes, int blockSize, bool isPixelwise)
        {
            BlockSize = blockSize;
            int maxX = bytes.GetUpperBound(1) + 1;
            int maxY = bytes.GetUpperBound(0) + 1;
			double[,] filterX = new double[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
			double[,] filterY = new double[,] { { -1, -2, -1 }, { 0, 0, 0 }, { 1, 2, 1 } };
			double[,] doubleBytes = new double[maxY, maxX];
			for (int row = 0; row < maxY; row++)
			{
				for (int column = 0; column < maxX; column++)
				{
					doubleBytes[row, column] = (double)bytes[row, column];
				}
			}
			// градиенты
			double[,] Gx = ConvolutionHelper.Convolve(doubleBytes, filterX);
			double[,] Gy = ConvolutionHelper.Convolve(doubleBytes, filterY);
			// разделение на блоки
			this._blocks = new Block[(int)Math.Floor((float)(maxY / BlockSize)), (int)Math.Floor((float)(maxX / BlockSize))];
			for (int row = 0; row < _blocks.GetUpperBound(0) + 1; row++)
			{
				for (int column = 0; column < _blocks.GetUpperBound(1) + 1; column++)
				{
					_blocks[row, column] = new Block(BlockSize, Gx, Gy, row * BlockSize, column * BlockSize);
				}
			}
        }

        public OrientationField(int[,] bytes):this(bytes, DefaultSize)
        {
        }

		public OrientationField(int[,] bytes, int blockSize): this(bytes, blockSize, false)
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

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace CUDAFingerprinting.Common
{
	public class PixelwiseOrientationField
	{
		private Block _block;
		public const int DefaultSize = 16;
		private double[,] _orientation;
		// property
		public Block Block
		{
			get
			{
				return _block;
			}
		}
		public int BlockSize
		{
			get;
			private set;
		}
		public double[,] Orientation
		{
			get
			{
				return _orientation;
			}
		}

		public PixelwiseOrientationField(int[,] bytes, int blockSize)
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

			// рассчет направления для каждого пикселя
			_orientation = new double[maxY, maxX];
			// только один блок - персональный для каждого пикселя
			for (int centerRow = 0; centerRow < maxY; centerRow++)
			{
				for (int centerColumn = 0; centerColumn < maxX; centerColumn++)
				{
					_block = new Block(_orientation, BlockSize, Gx, Gy, centerRow, centerColumn);
				}
			}
		}

		public double GetOrientation(int x, int y)                  // метод, определяющий по входным координатам (х, у) поле напрваления в этой точке
		{
			return this._orientation[y, x];//In my opinion, ther was a mistake: 'y' stands for rows, and 'x' stands for columns, right?
		}

	
	}
}

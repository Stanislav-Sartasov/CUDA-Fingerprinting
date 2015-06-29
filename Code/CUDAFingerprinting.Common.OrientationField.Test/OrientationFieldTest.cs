using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.OrientationField;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace CUDAFingerprinting.Common.OrientationField.Test
{
	[TestClass]
	class OrientationFieldTest
	{
		[TestMethod]
		public static void DrawLine(Bitmap image, int x1, int y1, int x2, double angle)				// рисует прямую линию на изображении от точки (x1, y1) до точки (x2, y2)
		{
			int length = x2 - x1;
			double slope = Math.Tan(angle);
			for (int x = x1; x < x2; x++)
			{
				int y = (int)(x * slope);
				if (y >= 0 && y < image.Height)
				{
					image.SetPixel(x, y - 9000, Color.Black);
				}
			}
		}

		[TestMethod]
		public static void Visualization(Bitmap sourceImage, CUDAFingerprinting.OrientationField.OrientationField Field)		// визуализирует поле направлений в картинку, где на каждом блоке 16х16 находится отрезок, наклоненный под соответствующим углом
		{
			const int SIZE = 16;
			Bitmap orientedImage = new Bitmap(sourceImage.Width, sourceImage.Height);
			Graphics g = Graphics.FromImage(orientedImage);
			g.Clear(Color.White);

			// перебираем все блоки 16х16
			for (int row = 0; row < SIZE; row++)
			{
				for (int column = 0; column < SIZE; column++)
				{
					// в каждом блоке получаем направление и строим отрезок
					double angle = Field.Blocks[row, column].Orientation;
					if (angle > 0)			// прямая возрастает
					{
						// начинаем строить линию с точки (SIZE * column, SIZE * row + SIZE - 1)  -- с нижней левой точки блока
						DrawLine(orientedImage, SIZE * column, SIZE * row + SIZE - 1, SIZE * (column + 1), angle);
						orientedImage.Save(@"Resources\1_1_oriented.jpg");

					}
					else if (angle < 0)		// прямая убывает
					{
						// начинаем строить линию с точки (SIZE * column, SIZE * row)  -- с верхней левой точки блока
						DrawLine(orientedImage, SIZE * column, SIZE * row, SIZE * (column + 1), angle);
						orientedImage.Save(@"Resources\1_1_oriented.jpg");

					}
					else					// прямая горизонтальна
					{
						// начинаем строить линию с точки (SIZE * column, SIZE * row + SIZE / 2)
						DrawLine(orientedImage, SIZE * column, SIZE * row + SIZE / 2, SIZE * (column + 1), angle);
						orientedImage.Save(@"Resources\1_1_oriented.jpg");

					}				
				}
			}
			orientedImage.Save(@"Resources\1_1_oriented.jpg");
		}

		[TestMethod]
		public void MainMethod()
		{
			Bitmap image;
			try
			{  // Retrieve the image.
				image = new Bitmap(@"Resources\1_1.tif");
			}
			catch (ArgumentException)
			{
				Console.WriteLine("There was an error. Check the path to the image file.");
				image = null;
			}
			int[,] bytes = new int[image.Width, image.Height];
			ImageHelper.SaveIntArray(bytes, @"Resources\1_1.tif");

			CUDAFingerprinting.OrientationField.OrientationField FingerPrint = new CUDAFingerprinting.OrientationField.OrientationField(bytes);
			for (int x = 0; x < image.Width; x += 16)
			{
				Console.WriteLine(x / 16 + "\t" + FingerPrint.GetOrientation(x, 0) * 180 / Math.PI);
			}

			Visualization(image, FingerPrint);
		}
	}
}

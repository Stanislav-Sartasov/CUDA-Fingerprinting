using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using CUDAFingerprinting.Common;
using CUDAFingerprinting.OrientationField;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing;

namespace CUDAFingerprinting.Common.OrientationField.Test
{
	[TestClass]
	public class VisualizationTest
	{
		public VisualizationTest()
		{	
		}

		public static int[,] ConvertBitmapToByteArray(Bitmap image)
		{
			int[,] result = new int[image.Width, image.Height];
			if (image != null)
			{
				for (int x = 0; x < image.Width; x++)
				{
					for (int y = 0; y < image.Height; y++)
					{
						result[x, y] = image.GetPixel(x, y).R;
					}
				}
			}
			return result;
		}

		public static void Visualization(Bitmap sourceImage, CUDAFingerprinting.OrientationField.OrientationField Field)		// визуализирует поле направлений в картинку, где на каждом блоке 16х16 находится отрезок, наклоненный под соответствующим углом
		{
			const int SIZE = 16;
			Bitmap orientedImage = new Bitmap(sourceImage.Width, sourceImage.Height);
			Graphics g = Graphics.FromImage(orientedImage);
			g.Clear(Color.White);

			// перебираем все блоки 16х16
			for (int row = 0; row < Field.Blocks.GetUpperBound(0) + 1; row++)
			{
				for (int column = 0; column < Field.Blocks.GetUpperBound(1) + 1; column++)
				{
					// в каждом блоке получаем направление и строим отрезок
					double angle = Field.Blocks[row, column].Orientation;
					int x1 = SIZE * column;
					int x2 = SIZE * (column + 1);
					int y1, y2;
					if (angle == Math.PI / 2) // прямая ~ вертикальна
					{
						// начинаем строить линию с точки (SIZE * column + SIZE / 2, SIZE * row)
						x1 = x2 = SIZE * column + SIZE / 2;
						y1 = SIZE * row;
						y2 = SIZE * (row + 1);
						g.DrawLine(new Pen(Color.Black), new Point(x1, y1), new Point(x2, y2));
						//orientedImage.Save(@"1_1_oriented.jpg");
						continue;
					}
					int dy = (int)((x2 - x1) * Math.Tan(angle));
					if (dy == 0)					// прямая ~ горизонтальна
					{
						// начинаем строить линию с точки (SIZE * column, SIZE * row + SIZE / 2)
						y1 = SIZE * row + SIZE / 2;
						y2 = SIZE * row + SIZE / 2;
						g.DrawLine(new Pen(Color.Black), new Point(x1, y1), new Point(x2, y2));
						//orientedImage.Save(@"1_1_oriented.jpg");
						continue;
					}
					if (angle > 0)			// прямая убывает
					{
						// начинаем строить линию с точки (SIZE * column, SIZE * row + SIZE - 1)  -- с верхней левой точки блока
						y1 = SIZE * row;
						if (Math.Abs(dy) >= Field.Blocks[row, column].Size)
						{
							y2 = Field.Blocks[row, column].Size - 1 + y1;
						}
						else
						{
							y2 = dy + y1;
						}
						g.DrawLine(new Pen(Color.Blue), new Point(x1, y1), new Point(x2, y2));
						//orientedImage.Save(@"1_1_oriented.jpg");
						continue;
					}
					if (angle < 0)		// прямая возрастает
					{
						// начинаем строить линию с точки (SIZE * column, SIZE * row)  -- с нижней левой точки блока
						y1 = SIZE * (row + 1) - 1;
						if (Math.Abs(dy) >= Field.Blocks[row, column].Size)
						{
							y2 = y1 - Field.Blocks[row, column].Size + 1;
						}
						else
						{
							y2 = y1 + dy; // dy < 0
						}
						g.DrawLine(new Pen(Color.Green), new Point(x1, y1), new Point(x2, y2)); 
						//orientedImage.Save(@"1_1_oriented.jpg");
						continue;
					}
					
				}
			}
			orientedImage.Save(@"1_1_oriented.jpg");
		}

		[TestMethod]
		public void TestMethod()
		{
			VisualizationTest test = new VisualizationTest();
			Bitmap image;
			int[,] bytes = null;
			try
			{  // Retrieve the image.
				image = Resources._1_1;
				// bytes = new int[image.Width, image.Height];
				// ImageHelper.SaveIntArray(bytes, @"Resources\1_1.jpg");
				bytes = ConvertBitmapToByteArray(image);
			}
			catch (ArgumentException)
			{
				Console.WriteLine("There was an error. Check the path to the image file.");
				image = null;
			}

			CUDAFingerprinting.OrientationField.OrientationField FingerPrint = new CUDAFingerprinting.OrientationField.OrientationField(bytes);
			for (int x = 0; x + 16 < image.Width; x += 16)
			{
				Console.WriteLine("x = " + x);
				for (int y = 0; y + 16 < image.Height; y += 16)
				{
					Console.Write("y = " + y + " " + (int)(FingerPrint.GetOrientation(x, y) * 180 / Math.PI) + "\t");
				}
			}

			 Visualization(image, FingerPrint);
		}
	}
}

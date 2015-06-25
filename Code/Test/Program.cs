using CUDAFingerprinting.Common;
using CUDAFingerprinting.OrientationField;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

namespace Test
{
	class Program
	{
		public static byte[,] ConvertBitmapToByteArray(Bitmap image)
		{
			byte[,] result = new byte[image.Width, image.Height];
			if (image != null)
			{
				for (int x = 0; x < image.Width; x++)
				{
					for (int y = 0; y < image.Height; y++)
					{
						result[x, y] = image.GetPixel(x, y).R;
						// Console.Write(result[x, y] + "\t");
					}
				}
			}
			return result;
		}

		public static void Visualization(Bitmap sourceImage, OrientationField Field)		// визуализирует поле направлений в картинку, где на каждом блоке 16х16 находится отрезок, наклоненный под соответствующим углом
		{
			const int SIZE = 16;
			Bitmap orientedImage = new Bitmap(sourceImage.Width, sourceImage.Height);
			// перебираем все блоки 16х16
			for (int row = 0; row < SIZE; row++)
			{
				for (int column = 0; column < SIZE; column++)
				{
					// в каждом блоке получаем направление и строим отрезок
					double angle = Field.blocks[row, column].orientation;
					if (angle > 0)			// прямая возрастает
					{

					}
					//else if()
				}
			}
			orientedImage.Save(@"D:\Education\CUDA Fingerprinting 2\Db\Db2_a\oriented1_1.tif");
		}

		static void Main(string[] args)
		{
			Bitmap image;
			try
			{  // Retrieve the image.
				image = new Bitmap(@"D:\Education\CUDA Fingerprinting 2\Db\Db2_a\1_1.tif");
			}
			catch (ArgumentException)
			{
				Console.WriteLine("There was an error. Check the path to the image file.");
				image = null;
			}

			byte[,] bytes = ConvertBitmapToByteArray(image);

			OrientationField FingerPrint = new OrientationField(bytes);
			for (int x = 0; x < image.Width; x += 16)
			{
				Console.WriteLine(x / 16 + "\t" + FingerPrint.GetOrientation(x, 0) * 180 / Math.PI);
			}

			Visualization(image, FingerPrint);

		}
	}
}

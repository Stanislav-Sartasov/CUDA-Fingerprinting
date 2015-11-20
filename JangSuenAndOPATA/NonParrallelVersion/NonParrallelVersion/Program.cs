using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading;
using System.Data;
using System.Drawing;
using System.Diagnostics;

namespace NonParrallelVersion
{
    class Program
    {
        private static Bitmap MatrixAsImage(int[,] Matrix, Bitmap Image)
        {
            for (int i = 0; i < Image.Width; i++)
            {
                for (int j = 0; j < Image.Height; j++)
                {
                    if (Matrix[i, j] == 1)
                        Image.SetPixel(i, j, Color.FromArgb(0, 0, 0));
                    else
                        Image.SetPixel(i, j, Color.FromArgb(255, 255, 255));
                }
            }
            return Image;
        }

        private static int[,] ImageAsMatrix(Bitmap Image)
        {
            int[,] Matrix = new int[Image.Width, Image.Height];
            for (int i = 0; i < Image.Width; i++)
            {
                for (int j = 0; j < Image.Height; j++)
                {
                    int color = (Image.GetPixel(i, j).R + Image.GetPixel(i, j).G + Image.GetPixel(i, j).B) / 3;
                    if (color < 255 / 2)
                    {
                        Matrix[i, j] = 1;
                    }
                    else Matrix[i, j] = 0;
                }
            }
            return Matrix;
        }

        static void Main(string[] args)
        {
            Console.WriteLine("Введите адрес картинки");
            string AddressRead = Console.ReadLine();

            Bitmap Image = new Bitmap(AddressRead);

            Console.WriteLine("Выберите алгоритм:\n1.Modified Jang-Suen;\n2.OPATA");
            int choose = Convert.ToInt32(Console.ReadLine());

            int[,] ImageMatrix = ImageAsMatrix(Image);
            int[,] newImageMatrix = new int[Image.Width, Image.Height];
            
            Stopwatch timer = new Stopwatch();
            timer.Start();
            
            if (choose == 1)
            {
                ModifiedJangSuen A = new ModifiedJangSuen();
                newImageMatrix = A.JangSuen(ImageMatrix, Image.Width, Image.Height);
            }
            if (choose == 2)
            {
                AlgorythmOPATA OP = new AlgorythmOPATA();
                newImageMatrix = OP.OPATA(ImageMatrix, Image.Width, Image.Height);
            }

            timer.Stop();
            Console.WriteLine("Время работы выбранного алгоритма: {0} ms", timer.ElapsedMilliseconds);
            Console.WriteLine("Картинка успешно обработана. Введите адрес для сохранения");
            string AddressWrite = Console.ReadLine();

            Image = MatrixAsImage(newImageMatrix, Image);
            Image.Save(AddressWrite);

            Console.WriteLine("Success!");
            Console.ReadKey();
        }
    }
}

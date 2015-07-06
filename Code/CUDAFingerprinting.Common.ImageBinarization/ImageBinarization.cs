using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace CUDAFingerprinting.Common.ImageBinarization
{
    public static class ImageBinarization
    {
        public static Bitmap Binarizator(Bitmap src, int line)
        {
            Bitmap bmp = new Bitmap(src.Width, src.Height);
            for (int i = 0; i < bmp.Width; i++)
            {
                for (int j = 0; j < bmp.Height; j++)
                {
                    bmp.SetPixel(i, j, src.GetPixel(i, j).B < line ? Color.Black : Color.White);
                }
            }
            return bmp;
        }

        public static int[,] Binarizator2D(int[,] src, int line)
        {
            int srcWidth = src.GetLength(0);
            int srcHeight = src.GetLength(1);
            int[,] imgInt = new int[srcWidth, srcHeight];

            for (int i = 0; i < srcWidth; i++)
            {
                for (int j = 0; j < srcHeight; j++)
                {
                    imgInt[i, j] = src[i, j] < line ? 0 : 255;
                }
            }
            return imgInt;
        }

        public static double[,] Binarizator2D(double[,] src, int line)
        {
            int srcWidth = src.GetLength(0);
            int srcHeight = src.GetLength(1);
            double[,] imgDouble = new double[srcWidth, srcHeight];
            for (int i = 0; i < srcWidth; i++)
            {
                for (int j = 0; j < srcHeight; j++)
                {
                    imgDouble[i, j] = src[i, j] < line ? 0 : 255;
                }
            }
            return imgDouble;
        }
    }
}

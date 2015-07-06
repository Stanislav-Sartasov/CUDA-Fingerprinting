using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;


namespace CUDAFingerprinting.Common.Matrix
{
    public class Matrix
    {
        private const int edge = 100;
        public Matrix() 
        {
        }
        public void calculatingAverageColor(String filename)
        {
            Bitmap pic = new Bitmap(filename);

            int[,] Gx = new int[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            int[,] Gy = new int[,] { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };
            int[,] Matrix = new int[pic.Height, pic.Width];

            int height = pic.Height;
            int width = pic.Width;

            //Using Sobel Operator
            for (int x = 1; x < pic.Height - 1; x++)
            {
                for (int y = 1; y < pic.Width - 1; y++)
                {
                    int sumX = pic.GetPixel(x - 1, y - 1).R * Gx[0, 0] + pic.GetPixel(x + 1, y - 1).R * Gx[0, 2] +
                                    pic.GetPixel(x - 1, y).R * Gx[11, 0] + pic.GetPixel(x + 1, y).R * Gx[1, 2] +
                                    pic.GetPixel(x - 1, y + 1).R * Gx[2, 0] + pic.GetPixel(x + 1, y + 1).R * Gx[2, 2];
                    int sumY = pic.GetPixel(x - 1, y - 1).R * Gy[0, 0] + pic.GetPixel(x, y - 1).R * Gy[0, 1] + pic.GetPixel(x + 1, y - 1).R * Gy[0, 2] +
                        pic.GetPixel(x - 1, y + 1).R * Gy[2, 0] + pic.GetPixel(x, y + 1).R * Gy[2, 1] + pic.GetPixel(x + 1, y + 1).R * Gy[2, 2];
                    int sqrtXY = (int)System.Math.Sqrt(sumX * sumX + sumY * sumY);

                    pic.SetPixel(x, y, Color.FromArgb(sqrtXY, sqrtXY, sqrtXY));
                }
            }

            //Creating Matrix with '1' for white and '0' for black
            for (int x = 0; x < width; x = x + 16)
            {
                for (int y = 1; y < height; y = y + 16)
                {
                    Rectangle cloneRect = new Rectangle(x, y, 16, 16);
                    PixelFormat format = pic.PixelFormat;
                    Bitmap block = pic.Clone(cloneRect, format);

                    int averageColor = 0;

                    for (int i = 0; i < 16; i++)
                    {
                        for (int j = 0; j < 16; j++)
                        {
                            Color pix = block.GetPixel(i, j);
                            averageColor += pix.R;
                        }
                    }

                    averageColor /= 256;

                    if (averageColor >= edge)
                    {
                        for (int i = 0; i < 16; i++)
                        {
                            for (int j = 0; j < 16; j++)
                            {
                                if (block.GetPixel(i, j).R >= edge)
                                {
                                    Matrix[x + i, y + j] = 1;
                                }
                                else
                                {
                                    Matrix[x + i, y + j] = 0;
                                }
                            }
                        }
                    }
                }
            }

            //Creating Black-White Bitmap on the basis of Matrix

            Bitmap newPic = new Bitmap(width, height);

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    if (Matrix[x, y] == 1)
                    {
                        newPic.SetPixel(x, y, Color.White);
                    }
                    else
                    {
                        newPic.SetPixel(x, y, Color.Black);
                    }
                }
            }

            newPic.Save("newPic.bmp");
        }
    }
}
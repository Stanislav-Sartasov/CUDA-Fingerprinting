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

        private int[,] matrix;
        private Bitmap pic;

        public Matrix(Bitmap picture)
        {
            matrix = new int[picture.Width, picture.Height];
            pic = new Bitmap(picture);
        }

        public Bitmap SobelFilter()
        {
            int width = pic.Width;
            int height = pic.Height; 

            int[,] Gx = new int[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            int[,] Gy = new int[,] { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            Bitmap SFPic = new Bitmap(pic);

            //Using Sobel Operator
            for (int x = 1; x < width - 1; x++)
            {
                for (int y = 1; y < height - 1; y++)
                {
                    int sumX = pic.GetPixel(x - 1, y - 1).R * Gx[0, 0] + pic.GetPixel(x + 1, y - 1).R * Gx[0, 2] +
                                    pic.GetPixel(x - 1, y).R * Gx[1, 0] + pic.GetPixel(x + 1, y).R * Gx[1, 2] +
                                    pic.GetPixel(x - 1, y + 1).R * Gx[2, 0] + pic.GetPixel(x + 1, y + 1).R * Gx[2, 2];
                    int sumY = pic.GetPixel(x - 1, y - 1).R * Gy[0, 0] + pic.GetPixel(x, y - 1).R * Gy[0, 1] + pic.GetPixel(x + 1, y - 1).R * Gy[0, 2] +
                        pic.GetPixel(x - 1, y + 1).R * Gy[2, 0] + pic.GetPixel(x, y + 1).R * Gy[2, 1] + pic.GetPixel(x + 1, y + 1).R * Gy[2, 2];
                    int sqrtXY = (int)System.Math.Sqrt(sumX * sumX + sumY * sumY);

                    if (sqrtXY > 255)
                    {
                        sqrtXY = 255;
                    }
                    if (sqrtXY < 0)
                    {
                        sqrtXY = 0;
                    }

                    SFPic.SetPixel(x, y, Color.FromArgb(sqrtXY, sqrtXY, sqrtXY));
                }
            }

            return SFPic;
        }

        public void MatrixMaking()
        {
            int width = pic.Width;
            int height = pic.Height; 

            //Creating Matrix with '1' for white and '0' for black
            for (int x = 0; x < width; x = x + 16)
            {
                for (int y = 0; y < height - 16; y = y + 16)
                {
                    Rectangle cloneRect = new Rectangle(x, y, 16, 16);
                    PixelFormat format = pic.PixelFormat;
                    Bitmap block = pic.Clone(cloneRect, format);

                    int averageColor = 0;

                    for (int i = 0; i < 16; i++)
                    {
                        for (int j = 0; j < 16; j++)
                        {
                            Color pixColor = block.GetPixel(i, j);
                            averageColor += pixColor.R;
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
                                    matrix[x + i, y + j] = 1;
                                }
                                else
                                {
                                    matrix[x + i, y + j] = 0;
                                }
                            }
                        }
                    }
                }
            }

            //Processing the bottom of the image
            for (int x = 0; x < width; x = x + 16)
            {

                Rectangle cloneRect = new Rectangle(x, 352, 16, 12);
                PixelFormat format = pic.PixelFormat;
                Bitmap block = pic.Clone(cloneRect, format);

                int averageColor = 0;

                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 12; j++)
                    {
                        Color pixColor = block.GetPixel(i, j);
                        averageColor += pixColor.R;
                    }
                }

                averageColor /= 192;

                if (averageColor >= edge)
                {
                    for (int i = 0; i < 16; i++)
                    {
                        for (int j = 0; j < 12; j++)
                        {
                            if (block.GetPixel(i, j).R >= edge)
                            {
                                matrix[x + i, 352 + j] = 1;
                            }
                            else
                            {
                                matrix[x + i, 352 + j] = 0;
                            }
                        }
                    }
                }
            }
        }

        public Bitmap BWPicture()
        {
            int width = pic.Width;
            int height = pic.Height; 

            //Creating Black-White Bitmap on the basis of Matrix

            Bitmap newPic = new Bitmap(width, height);

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    if (matrix[x, y] == 1)
                    {
                        newPic.SetPixel(x, y, Color.White);
                    }
                    else
                    {
                        newPic.SetPixel(x, y, Color.Black);
                    }
                }
            }

            return newPic;
        }
    }
}
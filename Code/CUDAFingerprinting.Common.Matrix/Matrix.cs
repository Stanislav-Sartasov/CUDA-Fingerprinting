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
        private const int edge = 100; //Color edge for block
        private const int pixEdge = 120; //Color edge for each pixel (for more effective setting)

        private double[,] matrix; //Saves results of using Sobel filter
        private Bitmap pic; //Saves a source picture

        public Matrix(Bitmap picture)
        {
            matrix = new double[picture.Width, picture.Height];
            pic = new Bitmap(picture);
        }

        public void SobelFilter()
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
                    double sqrtXY = System.Math.Sqrt(sumX * sumX + sumY * sumY);

                    matrix[x, y] = sqrtXY;
                }
            }
        }

        public int[,] MatrixMaking()
        {
            int width = pic.Width;
            int height = pic.Height;

            int[,] intMatrix = new int [width, height];

            //Creating Matrix with '1' for white and '0' for black

            for (int x = 0; x <= width - 16; x = x + 16)
            {
                for (int y = 0; y <= height - 16; y = y + 16)
                {
                    double averageColor = 0;

                    for (int i = 0; i < 16; i++)
                    {
                        for (int j = 0; j < 16; j++)
                        {

                            averageColor += matrix[x + i, y + j];
                        }
                    }

                    averageColor /= 256;

                    if (averageColor >= edge)
                    {
                        for (int i = 0; i < 16; i++)
                        {
                            for (int j = 0; j < 16; j++)
                            {
                                if (matrix[x + i, y + j] >= pixEdge)
                                {
                                    intMatrix[x + i, y + j] = 1;
                                }
                                else
                                {
                                    intMatrix[x + i, y + j] = 0;
                                }
                            }
                        }
                    }
                }
            }

            //Processing the bottom of the image
            if (height % 16 != 0)
            {
                for (int x = 0; x <= width - 16; x = x + 16)
                {
                    /*Rectangle cloneRect = new Rectangle(x, height - (height % 16), 16, height % 16);
                    PixelFormat format = pic.PixelFormat;
                    Bitmap block = pic.Clone(cloneRect, format);*/

                    double averageColor = 0;

                    for (int i = 0; i < 16; i++)
                    {
                        for (int j = 0; j < height % 16; j++)
                        {
                            //Color pixColor = block.GetPixel(i, j);
                            averageColor += matrix[x + i, height - (height % 16) + j];
                        }
                    }

                    averageColor /= (16 * (height % 16));

                    if (averageColor >= edge)
                    {
                        for (int i = 0; i < 16; i++)
                        {
                            for (int j = 0; j < height % 16; j++)
                            {
                                if (matrix[x + i, height - (height % 16) + j] >= pixEdge)
                                {
                                    intMatrix[x + i, height - (height % 16) + j] = 1;
                                }
                                else
                                {
                                    intMatrix[x + i, height - (height % 16) + j] = 0;
                                }
                            }
                        }
                    }
                }
            }

            //Processing the right side of the image
            if (width % 16 != 0)
            {
                for (int y = 0; y <= height - 16; y = y + 16)
                {
                    /*Rectangle cloneRect = new Rectangle(width - (width % 16), y, width % 16, 16);
                    PixelFormat format = pic.PixelFormat;
                    Bitmap block = pic.Clone(cloneRect, format);*/

                    double averageColor = 0;

                    for (int i = 0; i < width % 16; i++)
                    {
                        for (int j = 0; j < 16; j++)
                        {
                            //Color pixColor = block.GetPixel(i, j);
                            averageColor += matrix[width - (width % 16) + i, y + j];
                        }
                    }

                    averageColor /= (16 * (width % 16));

                    if (averageColor >= edge)
                    {
                        for (int i = 0; i < width % 16; i++)
                        {
                            for (int j = 0; j < 16; j++)
                            {
                                if (matrix[width - (width % 16) + i, y + j] >= pixEdge)
                                {
                                    intMatrix[width - (width % 16) + i, y + j] = 1;
                                }
                                else
                                {
                                    intMatrix[width - (width % 16) + i, y + j] = 0;
                                }
                            }
                        }
                    }
                }
            }

            //Processing the right bottom square of the image
            if (width % 16 != 0 && height % 16 != 0)
            {
                /*Rectangle cloneRect2 = new Rectangle(width - (width % 16), height - (height % 16), width % 16, height % 16);
                PixelFormat format2 = pic.PixelFormat;
                Bitmap block2 = pic.Clone(cloneRect2, format2);*/

                double averageColor2 = 0;

                for (int i = 0; i < 16; i++)
                {
                    for (int j = 0; j < 16; j++)
                    {
                        //Color pixColor = block2.GetPixel(i, j);
                        averageColor2 += matrix[width - (width % 16) + i, height - (height % 16) + j];
                    }
                }

                averageColor2 /= ((width % 16) * (height % 16));

                if (averageColor2 >= edge)
                {
                    for (int i = 0; i < 16; i++)
                    {
                        for (int j = 0; j < 16; j++)
                        {
                            if (matrix[width - (width % 16) + i, height - (height % 16) + j] >= pixEdge)
                            {
                                intMatrix[width - (width % 16) + i, height - (height % 16) + j] = 1;
                            }
                            else
                            {
                                intMatrix[width - (width % 16) + i, height - (height % 16) + j] = 0;
                            }
                        }
                    }
                }
            }

            return intMatrix;
        }

        public Bitmap BWPicture(int[,] intMatrix)
        {
            int width = pic.Width;
            int height = pic.Height; 

            //Creating Black-White Bitmap on the basis of Matrix

            Bitmap newPic = new Bitmap(width, height);

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    if (intMatrix[x, y] == 1)
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
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.FetureExtraction
{
    public class Segmentator
    {
        private const int edge = 50; //Color edge for block
        private const int pixEdge = 150; //Color edge for each pixel (for more effective setting)

        private float[,] matrix; //Saves results of using Sobel filter
        private double[,] pic; //Saves a source picture
        private int width;
        private int height;

        public  Segmentator(Bitmap picture)
        {
            matrix = new float[picture.Width, picture.Height];
            pic = ImageHelper.LoadImage(picture);
            width = picture.Width;
            height = picture.Height;
        }

        public float[,] SobelFilter()
        {
            int[,] filterX = new int[,] { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
            int[,] filterY = new int[,] { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

            //Using Sobel Operator
            for (int x = 1; x < width - 1; x++)
            {
                for (int y = 1; y < height - 1; y++)
                {
                    double sumX = pic[height - 1 - y + 1, x - 1] * filterX[0, 0] + pic[height - 1 - y + 1, x + 1] * filterX[0, 2] +
                                    pic[height - 1 - y, x - 1] * filterX[1, 0] + pic[height - 1 - y, x + 1] * filterX[1, 2] +
                                    pic[height - 1 - y - 1, x - 1] * filterX[2, 0] + pic[height - 1 - y - 1, x + 1] * filterX[2, 2];
                    double sumY = pic[height - 1 - y + 1, x - 1] * filterY[0, 0] + pic[height - 1 - y, x] * filterY[0, 1] + pic[height - 1 - y + 1, x + 1] * filterY[0, 2] +
                        pic[height - 1 - y - 1, x - 1] * filterY[2, 0] + pic[height - 1 - y - 1, x] * filterY[2, 1] + pic[height - 1 - y - 1, x + 1] * filterY[2, 2];
                    double sqrtXY = System.Math.Sqrt(sumX * sumX + sumY * sumY);

                    sqrtXY = sqrtXY > 255 ? 255 : sqrtXY;

                    matrix[x, y] = (float)sqrtXY;
                }
            }
            return matrix;
        }

        public byte[,] Segmentate()
        {
            byte[,] byteMatrix = new byte[width, height];

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
                                    byteMatrix[x + i, y + j] = 1;
                                }
                                else
                                {
                                    byteMatrix[x + i, y + j] = 0;
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
                                    byteMatrix[x + i, height - (height % 16) + j] = 1;
                                }
                                else
                                {
                                    byteMatrix[x + i, height - (height % 16) + j] = 0;
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
                                    byteMatrix[width - (width % 16) + i, y + j] = 1;
                                }
                                else
                                {
                                    byteMatrix[width - (width % 16) + i, y + j] = 0;
                                }
                            }
                        }
                    }
                }
            }

            //Processing the right bottom square of the image
            if (width % 16 != 0 && height % 16 != 0)
            {
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
                                byteMatrix[width - (width % 16) + i, height - (height % 16) + j] = 1;
                            }
                            else
                            {
                                byteMatrix[width - (width % 16) + i, height - (height % 16) + j] = 0;
                            }
                        }
                    }
                }
            }

            return byteMatrix;
        }

        public Bitmap MakeBitmap (byte[,] byteMatrix)
        {
            Bitmap bmp = new Bitmap(width, height);

            for (int x = 0; x < width; ++x)
            {
                for (int y = 0; y < height; ++y)
                {
                    if (byteMatrix[x, y] == 1)
                    {
                        bmp.SetPixel(x, y, Color.White);
                    }
                    else
                    {
                        bmp.SetPixel(x, y, Color.Black);
                    }
                }
            }

            return bmp;
        }

        public void SaveSegmentation(Bitmap bmp, string filename)
        {
            bmp.Save(filename);
        }
    }
}
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;
using CUDAFingerprinting.Common;

namespace CUDAFingerprinting.Common.HarrisSegmentation
{
    public class HarrisSegmentation
    {
        private const int strenght = 300;

        private double[,] pic; //Saves a source picture
        int width;
        int height;

        public HarrisSegmentation(Bitmap picture)
        {
            pic = ImageHelper.LoadImage(picture);
            width = picture.Width;
            height = picture.Height;
        }

        public double[,] GaussFilter()
        {
            double[,] filterX = new double[,] { { -3, 0, 3 }, { -10, 0, 10 }, { -3, 0, 3 } };
            double[,] filterY = new double[,] { { 3, 10, 3 }, { 0, 0, 0 }, { -3, -10, -3} };

            double[,] Gx = ConvolutionHelper.Convolve(pic, filterX);
            double[,] Gy = ConvolutionHelper.Convolve(pic, filterY);

            double[,] X2 = new double[height, width];
            double[,] Y2 = new double[height, width];
            double[,] XY = new double[height, width];

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    X2[i, j] = Gx[i, j] * Gx[i, j];
                    Y2[i, j] = Gy[i, j] * Gy[i, j];
                    XY[i, j] = Gx[i, j] * Gy[i, j];
                }
            }

            double sigma = 0.5;
            int size = KernelHelper.GetKernelSizeForGaussianSigma(sigma);
            double[,] w = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, sigma), size); //Gaussian filter

            double[,] A = ConvolutionHelper.Convolve(X2, w); 
            double[,] B = ConvolutionHelper.Convolve(Y2, w);
            double[,] C = ConvolutionHelper.Convolve(XY, w);

            double[,] R = new double[height, width];

            for (int i = 0; i < height; ++i)
            {
                for (int j = 0; j < width; ++j)
                {
                    double[,] M = new double[2, 2];
                    M[0, 0] = A[i, j];
                    M[0, 1] = C[i, j];
                    M[1, 0] = C[i, j];
                    M[1, 1] = B[i, j];

                    double Tr = M[0, 0] + M[1, 1];
                    double Det = M[0, 0] * M[1, 1] - M[1, 0] * M[0, 1];
                    
                    R[i, j] = Det / Tr;
                }
            }

            double[,] matrix = new double[width, height];
            for ( int i = 0; i < width; ++i )
            {
                for ( int j = 0; j < height; ++j)
                {
                    matrix[i, j] = R[height - 1 - j, i];
                }
            }

            return matrix;
        }

        public int[,] Segmentate(double[,] matrix)
        {
            int[,] byteMatrix = new int[width, height];

            for (int x = 0; x < width - 3; x = x + 3)
            {
                for (int y = 0; y < height - 3; y = y + 3)
                {
                    bool flag = false;

                    for (int i = 0; i < 3; ++i)
                    {
                        for (int j = 0; j < 3; ++j)
                        {
                            if (matrix[x + i, y + j] > strenght)
                            {
                                flag = true;
                            }
                        }
                    }

                    for (int i = 0; i < 3; ++i)
                    {
                        for (int j = 0; j < 3; ++j)
                        {
                            if ( flag )
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

            return byteMatrix;
        }

        public Bitmap MakeBitmap (int[,] byteMatrix)
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

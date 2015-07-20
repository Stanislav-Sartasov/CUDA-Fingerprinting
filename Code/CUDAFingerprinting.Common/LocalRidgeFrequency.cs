using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;


namespace CUDAFingerprinting.Common
{
    public static class LocalRidgeFrequency
    {
        private static double[] CalculateSignature(double[,] img, double angle, int x, int y, int w, int l)
        {
            double[] signature = new double[l];
            double angleSin = Math.Sin(angle);
            double angleCos = Math.Cos(angle);
            for (int k = 0; k < l; k++)
            {
                for (int d = 0; d < w; d++)
                {
                    int indX = (int) (x + (d - w/2) * angleCos + (k - l/2) * angleSin);
                    int indY = (int) (y + (d - w / 2) * angleSin + (l / 2 - k) * angleCos);
                    if ((indX < 0) || (indY < 0) || (indX >= img.GetLength(0)) || (indY >= img.GetLength(1))) 
                        continue;
                    signature[k] += img[indX, indY];
                }
                signature[k] /= w;
            }
            return signature;
        }

        private static double CalculateFrequencyPixel(double[,] img, double angle, int x, int y, int w, int l)
        {
            double[] signature = CalculateSignature(img, angle, x, y, w, l);

            int prevMin    = -1;
            int lengthsSum = 0;
            int summandNum = 0;
            
            for (int i = 1; i < signature.Length - 1; i++)
            {
                //In comparison below there has to be non-zero value so that we would be able to ignore minor irrelevant pits.
                // 0.1 was calculated using heuristic approach. 
                if ((signature[i - 1] - signature[i] > 0.1) && (signature[i + 1] - signature[i] > 0.1))
                {
                    if (prevMin != -1)
                    {
                        lengthsSum += i - prevMin;
                        summandNum++;
                        prevMin = i;
                    }
                    else
                    {
                        prevMin = i;
                    }
                }
            }
            double frequency = (double) summandNum/lengthsSum;
            if ((lengthsSum <= 0) || (frequency > 1.0/3.0) || (frequency < 0.04))
                frequency = -1;
            return frequency;
        }

        public static double[,] CalculateFrequency(double[,] img, double[,] orientationMatrix, int w = 16, int l = 32)
        {
            double[,] frequencyMatrix = new double[img.GetLength(0), img.GetLength(1)];
            for (int i = 0; i < img.GetLength(0); i ++)
            {
                for (int j = 0; j < img.GetLength(1); j ++)
                {
                    frequencyMatrix[i, j] = CalculateFrequencyPixel(img, orientationMatrix[i, j], i, j, w, l);
                }
            }
            return frequencyMatrix;
        }

        private static double Mu(double x)
        {
            if (x <= 0) return 0;
            return x;
        }
        private static double Delta(double x)
        {
            if (x <= 0) return 0;
            return 1;
        }

        public static bool InterpolateFrequency(this double[,] frequencyMatrix, int filterSize = 7, double sigma = 1, int w = 16)
        {
            bool needMoreInterpolationFlag = false;

            int height = frequencyMatrix.GetLength(0);
            int width  = frequencyMatrix.GetLength(1);

            for (int i = 0; i < height; i ++)
            {
                for (int j = 0; j < width; j ++)
                {
                    if (frequencyMatrix[i, j] == -1.0)
                    {
                        var gaussian = new Filter(filterSize, sigma);
                        int center = filterSize/2; //filter is always a square.
                        int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;
                        double numerator   = 0;
                        double denominator = 0;
                        for (int u = -upperCenter; u <= center; u++)
                        {
                            for (int v = -upperCenter; v <= center; v++)
                            {
                                int indexX = i + u * w;
                                int indexY = j + v * w;
                                if (indexX < 0) indexX = 0;
                                if (indexX >= height) indexX = height - 1;
                                if (indexY < 0) indexY = 0;
                                if (indexY >= width) indexY = width - 1;
                                numerator += gaussian.Matrix[center - u, center - v]*Mu(frequencyMatrix[indexX, indexY]);
                                denominator += gaussian.Matrix[center - u, center - v]*Delta(frequencyMatrix[indexX, indexY] + 1);
                            }
                        }
                        frequencyMatrix[i, j] = numerator/denominator;
                        if ((frequencyMatrix[i, j] != frequencyMatrix[i, j]) || (frequencyMatrix[i, j] > 1.0/3.0) ||
                            (frequencyMatrix[i, j] < 0.04))
                        {
                            frequencyMatrix[i, j] = -1;
                            needMoreInterpolationFlag = true;
                        }
                    }
                }
            }
            return needMoreInterpolationFlag;
        }

        public static void InterpolateToPerfecton(this double[,] frequencyMatrix, int filterSize = 7, double sigma = 1, int w = 16)
        {
            bool flag = InterpolateFrequency(frequencyMatrix, filterSize, sigma, w);
            while (flag) 
                flag = InterpolateFrequency(frequencyMatrix, filterSize, sigma, w);
        }

        public static double[,] FilterFrequencies(double[,] frequencyMatrix, int filterSize = 7, double sigma = 1, int w = 16)
        {
            var result = new double[frequencyMatrix.GetLength(0), frequencyMatrix.GetLength(1)];
            var lowPassFilter = new Filter(filterSize, sigma);
            lowPassFilter.Normalize();
            
            result = StrangeConvolve(frequencyMatrix, lowPassFilter.Matrix, w);
            return result;
        }

        //Yeah, I know I shouldn't do stuff like this, but I'll fix it. Later.
        public static double[,] StrangeConvolve(double[,] data, double[,] kernel, int w)
        {
            int X = data.GetLength(0);
            int Y = data.GetLength(1);

            int I = kernel.GetLength(0);
            int J = kernel.GetLength(1);

            var result = new double[X, Y];

            var centerI = I / 2;
            var centerJ = J / 2;
            int upperLimitI = (I & 1) == 0 ? centerI - 1 : centerI;
            int upperLimitJ = (J & 1) == 0 ? centerJ - 1 : centerJ;

            for (int x = 0; x < X; x++)
            {
                for (int y = 0; y < Y; y++)
                {
                    for (int i = -upperLimitI; i <= centerI; i++)
                    {
                        for (int j = -upperLimitJ; j <= centerJ; j++)
                        {
                            var indexX = x + i * w;
                            if (indexX < 0) indexX = 0;
                            if (indexX >= X) indexX = X - 1;
                            var indexY = y + j * w;
                            if (indexY < 0) indexY = 0;
                            if (indexY >= Y) indexY = Y - 1;
                            result[x, y] += kernel[centerI - i, centerJ - j] * data[indexX, indexY];
                        }
                    }
                }
            }
            return result;
        }
    }
}

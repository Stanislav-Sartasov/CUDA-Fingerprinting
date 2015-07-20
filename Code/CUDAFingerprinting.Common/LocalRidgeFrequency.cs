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

        private static double CalculateFrequencyBlock(double[,] img, double angle, int x, int y, int w, int l)
        {
            double[] signature = CalculateSignature(img, angle, x, y, w, l);

            int prevMin    = -1;
            int lengthsSum = 0;
            int summandNum = 0;
            
            for (int i = 1; i < signature.Length - 1; i++)
            {
                if ((signature[i - 1] - signature[i] > 0) && (signature[i + 1] - signature[i] > 0))
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
            double[,] frequencyMatrix = new double[img.GetLength(0) / w, img.GetLength(1) / w];
            //iterating by block centres:
            for (int i = w/2 - 1; i <= (img.GetLength(0)/w - 1)*w + (w/2 - 1); i += w)
            {
                for (int j = w / 2 - 1; j <= (img.GetLength(1) / w - 1) * w + (w / 2 - 1); j += w)
                {
                    int a = (i - (w/2 - 1))/w;
                    int b = (j - (w/2 - 1))/w;
                    int c;
                    if ((a == 10) && (b == 3))
                    {
                        c = 1;
                    }
                    frequencyMatrix[(i - (w / 2 - 1)) / w, (j - (w / 2 - 1)) / w] = CalculateFrequencyBlock(img, orientationMatrix[i, j], i, j, w, l);
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
        //public static double[,] InterpolateFrequencyOld(double[,] frequencyMatrix, int height, int width, int filterSize = 7, int w = 16)
        //{
        //    double[,] result = new double[frequencyMatrix.GetLength(0), frequencyMatrix.GetLength(1)];
        //    for (int i = w / 2 - 1; i <= (height / w - 1) * w + (w / 2 - 1); i += w)
        //    {
        //        for (int j = w / 2 - 1; j <= (width / w - 1) * w + (w / 2 - 1); j += w)
        //        {
        //            if (frequencyMatrix[(i - (w / 2 - 1)) / w, (j - (w / 2 - 1)) / w] != -1.0)
        //            {
        //                result[(i - (w/2 - 1))/w, (j - (w/2 - 1))/w] =
        //                    frequencyMatrix[(i - (w/2 - 1))/w, (j - (w/2 - 1))/w];
        //                //for (int u = i - (w/2 - 1); u < i + (w/2 + 1); u++)
        //                //{
        //                //    for (int v = j - (w/2 - 1); v < j + (w/2 + 1); v++)
        //                //    {
        //                //        result[u, v] = frequencyMatrix[i, j];
        //                //    }
        //                //}
        //            }
        //            else
        //            {
        //                var gaussian = new Filter(filterSize, 1);
        //                int center = filterSize / 2; //filter is always a square.
        //                int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;
        //                double numerator = 0;
        //                double denominator = 0;
        //                for (int u = -upperCenter; u <= center; u++)
        //                {
        //                    for (int v = -upperCenter; v <= center; v++)
        //                    {
        //                        int indexX = (i + u * w) / w;
        //                        int indexY = (j + v * w) / w;
        //                        if (indexX < 0) indexX = 0;
        //                        if (indexX >= frequencyMatrix.GetLength(0)) indexX = frequencyMatrix.GetLength(0) - 1;
        //                        if (indexY < 0) indexY = 0;
        //                        if (indexY >= frequencyMatrix.GetLength(1)) indexY = frequencyMatrix.GetLength(1) - 1;
        //                        numerator +=  gaussian.Matrix[center - u, center - v] * Mu(frequencyMatrix[indexX, indexY]);//Not sure this formula is correct
        //                        denominator += gaussian.Matrix[center - u, center - v] * Delta(frequencyMatrix[indexX, indexY] + 1);//Not sure this formula is correct
        //                    }
        //                }
        //                frequencyMatrix[(i - (w / 2 - 1)) / w, (j - (w / 2 - 1)) / w] = numerator/denominator;
        //                result[(i - (w/2 - 1))/w, (j - (w/2 - 1))/w] =
        //                    frequencyMatrix[(i - (w/2 - 1))/w, (j - (w/2 - 1))/w];
        //                //for (int u = i - (w / 2 - 1); u < i + (w / 2 + 1); u++)
        //                //{
        //                //    for (int v = j - (w / 2 - 1); v < j + (w / 2 + 1); v++)
        //                //    {
        //                //        result[u, v] = frequencyMatrix[i, j];
        //                //    }
        //                //}
        //            }
        //        }
        //    }
        //    return result;
        //}

        public static bool InterpolateFrequency(this double[,] frequencyMatrix, int filterSize = 7, int sigma = 1, int w = 16)
        {
            bool needMoreInterpolationFlag = false;

            int height = frequencyMatrix.GetLength(0);
            int width = frequencyMatrix.GetLength(1);

            for (int i = 0; i < height; i ++)
            {
                for (int j = 0; j < width; j ++)
                {
                    if (frequencyMatrix[i, j] == -1.0)
                    {
                        var gaussian = new Filter(filterSize, sigma);
                        int center = filterSize/2; //filter is always a square.
                        int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;
                        double numerator = 0;
                        double denominator = 0;
                        for (int u = -upperCenter; u <= center; u++)
                        {
                            for (int v = -upperCenter; v <= center; v++)
                            {
                                //In paper by Hong, Wan, Jain the indexes look different, but they're equivalent to those below.
                                int indexX = i + u;
                                int indexY = j + v;
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

        public static void InterpolateToPerfecton(this double[,] frequencyMatrix, int filterSize = 7, int w = 16)
        {
            bool flag = InterpolateFrequency(frequencyMatrix, filterSize, w);
            while (flag) 
                flag = InterpolateFrequency(frequencyMatrix, filterSize, w);
        }

        public static double[,] Filter(double[,] frequencyMatrix, int filterSize = 7, double sigma = 1, int w = 16)
        {
            var result = new double[frequencyMatrix.GetLength(0), frequencyMatrix.GetLength(1)];
            var lowPassFilter = new Filter(filterSize, sigma);
            lowPassFilter.Normalize();
            result = ConvolutionHelper.Convolve(frequencyMatrix, lowPassFilter.Matrix);
            return result;
        }
        public static double[,] GetFrequencyMatrixImageSize(double[,] frequencyMatrix, int imgHeight, int imgWidth, int w = 16)
        {
            double[,] result = new double[imgHeight, imgWidth];
            for (int i = 0; i < frequencyMatrix.GetLength(0); i++)
            {
                for (int j = 0; j < frequencyMatrix.GetLength(1); j++)
                {
                    double curFrequency = frequencyMatrix[i, j];
                    for (int u = 0; u < w; u++)
                    {
                        for (int v = 0; v < w; v++)
                        {
                            result[i*w + u, j*w + v] = curFrequency;
                        }
                    }
                }
            }
            return result;
        }
    }
}

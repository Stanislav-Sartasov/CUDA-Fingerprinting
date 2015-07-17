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
                if ((signature[i - 1] > signature[i]) && (signature[i + 1] > signature[i]))
                {
                    if (prevMin != -1)
                    {
                        lengthsSum += i - prevMin;
                        summandNum++;
                        prevMin = -1;
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
            for (int i = w/2 - 1; i <= (img.GetLength(0)/w - 1)*w + (w/2 - 1); i += w)
            {
                for (int j = w / 2 - 1; j <= (img.GetLength(1) / w - 1) * w + (w / 2 - 1); j += w)
                {
                    double frequency = CalculateFrequencyBlock(img, orientationMatrix[i, j], i, j, w, l);
                    for (int u = i - (w/2 - 1); u < i + (w / 2 + 1); u++)
                    {
                        for (int v = j - (w/2 - 1); v < j + (w/2 + 1); v++)
                        {
                            frequencyMatrix[u, v] = frequency;
                        }
                    }
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
        public static double[,] InterpolateFrequency(double[,] frequencyMatrix, int height, int width, int filterSize = 7, int w = 16)
        {
            double[,] result = new double[frequencyMatrix.GetLength(0), frequencyMatrix.GetLength(1)];
            for (int i = w / 2 - 1; i <= (height / w - 1) * w + (w / 2 - 1); i += w)
            {
                for (int j = w / 2 - 1; j <= (width / w - 1) * w + (w / 2 - 1); j += w)
                {
                    if (frequencyMatrix[i, j] != -1)
                    {
                        for (int u = i - (w/2 - 1); u < i + (w/2 + 1); u++)
                        {
                            for (int v = j - (w/2 - 1); v < j + (w/2 + 1); v++)
                            {
                                result[u, v] = frequencyMatrix[i, j];
                            }
                        }
                    }
                    else
                    {
                        var gaussian = new Filter(filterSize, 1);
                        int center = filterSize / 2; //filter is always a square.
                        int upperCenter = (filterSize & 1) == 0 ? center - 1 : center;
                        double numerator = 0;
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
                                numerator +=  gaussian.Matrix[center - u, center - v] * Mu(frequencyMatrix[indexX, indexY]);//Not sure this formula is correct
                                denominator += gaussian.Matrix[center - u, center - v] * Delta(frequencyMatrix[indexX, indexY] + 1);//Not sure this formula is correct
                            }
                        }
                        frequencyMatrix[i,j] = numerator/denominator;
                        for (int u = i - (w / 2 - 1); u < i + (w / 2 + 1); u++)
                        {
                            for (int v = j - (w / 2 - 1); v < j + (w / 2 + 1); v++)
                            {
                                result[u, v] = frequencyMatrix[i, j];
                            }
                        }
                    }
                }
            }
            return result;
        }
    }
}

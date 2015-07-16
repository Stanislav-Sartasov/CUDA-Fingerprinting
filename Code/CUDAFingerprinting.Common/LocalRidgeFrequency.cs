using System;
using System.Collections.Generic;
using System.Linq;
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
    }
}

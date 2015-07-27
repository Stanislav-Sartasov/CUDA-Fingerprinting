using System;

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
                //In comparison below there has to be non-zero value so that we would be able to ignore minor irrelevant pits of black.
                // 0.3 was calculated using heuristic approach. 
                if ((signature[i - 1] - signature[i] > 0.3) && (signature[i + 1] - signature[i] > 0.3))
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

        private static double[,] CalculateFrequency(double[,] img, double[,] orientationMatrix, int w, int l)
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

        private static bool InterpolateFrequency(this double[,] frequencyMatrix, int filterSize, double sigma, int w)
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

        private static void InterpolateToPerfecton(this double[,] frequencyMatrix, int filterSize, double sigma, int w)
        {
            bool flag = InterpolateFrequency(frequencyMatrix, filterSize, sigma, w);
            while (flag) 
                flag = InterpolateFrequency(frequencyMatrix, filterSize, sigma, w);
        }

        private static double[,] FilterFrequencies(double[,] frequencyMatrix, int filterSize, double sigma, int w)
        {
            var result = new double[frequencyMatrix.GetLength(0), frequencyMatrix.GetLength(1)];
            var lowPassFilter = new Filter(filterSize, sigma);
            lowPassFilter.Normalize();
            
            result = ConvolutionHelper.Convolve(frequencyMatrix, lowPassFilter.Matrix, w);
            return result;
        }

        public static double[,] GetFrequencies(double[,] img, double[,] orientMatrix, int interpolationFilterSize = 7, double interpolationFilterSigma = 1,
            int lowPassFilterSize = 19, double lowPassFilterSigma = 3, int w = 16, int l = 32)
        {
            var frequencyMatr = CalculateFrequency(img, orientMatrix, w, l);
            frequencyMatr.InterpolateToPerfecton(interpolationFilterSize, interpolationFilterSigma, w);
            var filtered = FilterFrequencies(frequencyMatr, lowPassFilterSize, lowPassFilterSigma, w);
            return filtered;
        }
    }
}

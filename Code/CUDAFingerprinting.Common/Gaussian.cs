using System;

namespace CUDAFingerprinting.Common
{
    public class Gaussian
    {
        public static double Gaussian1D(double x, double sigma)
        {
            var commonDenom = 2.0d * sigma * sigma;
            var denominator = sigma * Math.Sqrt(Math.PI * 2);
            var result = Math.Exp(-(x * x) / commonDenom) / denominator;
            return result;
        }

        public static double Gaussian2D(double x, double y, double sigma)
        {
            var commonDenom = 2.0d * sigma * sigma;
            var denominator = Math.PI * commonDenom;
            var result = Math.Exp(-(x * x + y * y) / commonDenom) / denominator;
            return result;
        }

        public static double Gaussian2D(double x, double y, double sigmaX, double sigmaY)
        {
            var value = -(x * x / (sigmaX * sigmaX) + y * y / (sigmaY / sigmaY)) / 2.0;
            var denominator = 2.0 * Math.PI * sigmaX * sigmaY;
            double gaus = Math.Exp(value) / denominator;

            return gaus;
        }
    }
    public class Filter
    {
        public double[,] Matrix;

        public Filter(int size, double sigma)
        {
            Matrix = new double[size, size];

            int center = size / 2;
            int upperCenter = (size & 1) == 0 ? center - 1 : center;

            for (int i = -upperCenter; i <= upperCenter; i++)
            {
                for (int j = -upperCenter; j <= upperCenter; j++)
                {
                    Matrix[center - i, center - j] = Gaussian.Gaussian2D(i, j, sigma);
                }
            }
        }

        public void Normalize()
        {
            double sum = 0;
            for (int i = 0; i < Matrix.GetLength(0); i++)
            {
                for (int j = 0; j < Matrix.GetLength(0); j++)
                {
                    sum += Matrix[i, j];
                }
            }
            for (int i = 0; i < Matrix.GetLength(0); i++)
            {
                for (int j = 0; j < Matrix.GetLength(0); j++)
                {
                    Matrix[i, j] /= sum;
                }
            }
        }
        public void WriteMatrix()
        {
            int size = Matrix.GetLength(0);

            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    Console.Write("| {0:##.####} ", Matrix[i, j]);
                }

                Console.WriteLine("|");
            }
        }
    }
}

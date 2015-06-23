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
}

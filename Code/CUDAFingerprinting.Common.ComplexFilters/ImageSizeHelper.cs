using System;
using System.Drawing;

namespace CUDAFingerprinting.Common.ComplexFilters
{
    public class ImageSizeHelper
    {
        public static double[,] Reduce2(double[,] source, double factor)
        {

            var smoothed = ConvolutionHelper.Convolve(source,
                                                      KernelHelper.MakeKernel(
                                                          (x, y) => Gaussian.Gaussian2D(x, y, factor / 2d * 0.75d), KernelHelper.GetKernelSizeForGaussianSigma(factor / 2d * 0.75d)));
            var result = new double[(int)(source.GetLength(0) / factor), (int)(source.GetLength(1) / factor)];
            Resize(smoothed, result, factor, (x, y) => Gaussian.Gaussian2D(x, y, factor / 2d * 0.75d));
            return result;
        }

        public static double[,] Expand2(double[,] source, double factor, Size requestedSize = default(Size))
        {
            double[,] result = requestedSize == default(Size)
                                   ? new double[(int)(source.GetLength(0) * factor), (int)(source.GetLength(1) * factor)]
                                   : new double[requestedSize.Width, requestedSize.Height];
            Resize(source, result, 1 / factor, (x, y) => Gaussian.Gaussian2D(x, y, factor / 2d * 0.75d));
            return result;
        }

        private static void Resize(double[,] source, double[,] result, double cellSize, Func<double, double, double> filterFunction)
        {
            for (int row = 0; row < result.GetLength(0); row++)
            {
                for (int column = 0; column < result.GetLength(1); column++)
                {
                    double x = cellSize * row;
                    double y = cellSize * column;

                    double sum = 0;
                    double filterSum = 0;

                    for (int xm = (int)x - 5; xm <= (int)x + 5; xm++)
                    {
                        if (xm < 0) continue;
                        if (xm >= source.GetLength(0)) break;
                        for (int ym = (int)y - 5; ym <= (int)y + 5; ym++)
                        {
                            if (ym < 0) continue;
                            if (ym >= source.GetLength(1)) break;
                            var filterValue = filterFunction(x - xm, y - ym);
                            filterSum += filterValue;
                            sum += source[xm, ym] * filterValue;
                        }
                    }
                    sum /= filterSum;
                    result[row, column] = sum;
                }
            }
        }
    }

}

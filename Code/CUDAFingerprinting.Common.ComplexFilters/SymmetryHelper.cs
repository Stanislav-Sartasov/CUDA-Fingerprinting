using System;
using System.Numerics;

namespace CUDAFingerprinting.Common.ComplexFilters
{
    public class SymmetryHelper
    {
        public static Complex[,] EstimateLS(double[,] l1, double Sigma1, double Sigma2)
        {
            var kernelX = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma1) * x, KernelHelper.GetKernelSizeForGaussianSigma(Sigma1));
            var resultX = ConvolutionHelper.Convolve(l1, kernelX);
            var kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma1) * -y, KernelHelper.GetKernelSizeForGaussianSigma(Sigma1));
            var resultY = ConvolutionHelper.Convolve(l1, kernelY);


            var preZ = KernelHelper.MakeComplexFromDouble(resultX, resultY);

            var z = preZ.Select2D(x => x * x);

            var kernel2 = KernelHelper.MakeComplexKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma2), (x, y) => 0,
                KernelHelper.GetKernelSizeForGaussianSigma(Sigma2));

            var I20 = ConvolutionHelper.ComplexConvolve(z, kernel2);

            var I11 = ConvolutionHelper.Convolve(z.Select2D(x => x.Magnitude), kernel2.Select2D(x => x.Real));

            Complex[,] LS = KernelHelper.Zip2D(I20, I11, (x, y) => x / y);

            return LS;
        }

        public static Complex[,] EstimatePS(double[,] img, double Sigma1, double Sigma2)
        {
            var z = GetSquaredDerectionField(img, Sigma1);

            var kernel2 =
                KernelHelper.MakeComplexKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma2) * x / (x == 0 && y == 0 ? 1 : Math.Sqrt(x * x + y * y)),
                                               (x, y) => Gaussian.Gaussian2D(x, y, Sigma2) * y / (x == 0 && y == 0 ? 1 : Math.Sqrt(x * x + y * y)),
                                               KernelHelper.GetKernelSizeForGaussianSigma(Sigma2));

            var I20 = ConvolutionHelper.ComplexConvolve(z, kernel2);
            return I20;
        }
        public static Complex[,] EsimateH2(double[,] img, double Sigma1, double Sigma2)
        {
            var z = GetSquaredDerectionField(img, Sigma1);
            var kernel2 = KernelHelper.MakeComplexKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma2) * x, (x, y) => Gaussian.Gaussian2D(x, y, Sigma2) * (-y), KernelHelper.GetKernelSizeForGaussianSigma(Sigma2));

            var I20 = ConvolutionHelper.ComplexConvolve(z, kernel2);
            return I20;
        }

        public static Complex[,] GetSquaredDerectionField(double[,] img, double Sigma1)
        {
            var kernelX = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma1) * x, KernelHelper.GetKernelSizeForGaussianSigma(Sigma1));
            var resultX = ConvolutionHelper.Convolve(img, kernelX);
            var kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, Sigma1) * -y, KernelHelper.GetKernelSizeForGaussianSigma(Sigma1));
            var resultY = ConvolutionHelper.Convolve(img, kernelY);

            var preZ = KernelHelper.MakeComplexFromDouble(resultX, resultY);

            var z = preZ.Select2D(x => x * x);
            return z;
        }
    }
}

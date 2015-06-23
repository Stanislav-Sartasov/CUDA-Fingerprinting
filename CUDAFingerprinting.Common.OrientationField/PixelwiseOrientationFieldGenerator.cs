using System;

namespace CUDAFingerprinting.Common.OrientationField
{
    public class PixelwiseOrientationFieldGenerator
    {

        public static double[,] GenerateOrientationField(int[,] bytes)
        {
            return GenerateOrientationField(bytes.Select2D(x => (double)x));
        }

        public static double[,] GenerateOrientationField(double[,] bytes)
        {
            double size = 1;

            double avSigma = 5;

            var kernelAv = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, avSigma),
                                                   KernelHelper.GetKernelSizeForGaussianSigma(avSigma));

            var kernelX = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, size)*x,
                                                  KernelHelper.GetKernelSizeForGaussianSigma(size));

            var dx = ConvolutionHelper.Convolve(bytes, kernelX);

            var kernelY = KernelHelper.MakeKernel((x, y) => Gaussian.Gaussian2D(x, y, size)*-y,
                                                  KernelHelper.GetKernelSizeForGaussianSigma(size));

            var dy = ConvolutionHelper.Convolve(bytes, kernelY);

            var Gxx = dx.Select2D(x => x*x);

            var Gxy = dx.Select2D((x, row, column) => x*dy[row, column]);

            var Gyy = dy.Select2D(x => x*x);

            Gxx = ConvolutionHelper.Convolve(Gxx, kernelAv);
            Gxy = ConvolutionHelper.Convolve(Gxy, kernelAv);
            Gyy = ConvolutionHelper.Convolve(Gyy, kernelAv);

            var angles = Gxx.Select2D((gxx, row, column) => 0.5*Math.Atan2( 2.0*Gxy[row, column],gxx - Gyy[row, column]));
            
            angles = angles.Select2D(angle => angle <= 0 ? angle + Math.PI/2 : angle - Math.PI/2);
            //ImageHelper.SaveFieldAbove(bytes, angles, "C:\\temp\\orField.png");
            //ImageHelper.SaveArray(angles, "C:\\temp\\angles.png");
            return angles;
        }
        //private static void Main()
        //{
        //    ImageHelper.SaveArray(GenerateOrientationField(ImageHelper.LoadImage("C:\\temp\\104_6.tif").Select2D(x => (int)x)), "C:\\temp\\of.png");
        //}
    }

        
}

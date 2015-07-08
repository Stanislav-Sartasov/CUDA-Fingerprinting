using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;


namespace CUDAFingerprinting.GPU.Normalization.Test
{
    [TestClass]
    public class NormalizationTests
    {
        [DllImport("CUDAFingerprinting.GPU.Normalization.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Normalize")]
        public static extern void Normalize(float[] source, float[] res, int imgWidth, int imgHeight, int bordMean, int bordVar);

        [TestMethod]
        public void NormalizationTest()
        {
            var bmp = Resources.SampleFinger1;
            float[,] array0 = ImageHelper.LoadImageToFloats(bmp);
            float[] array = array0.Make1D();
            float[] result = new float[bmp.Width * bmp.Height];
            Normalize(array, result, bmp.Width, bmp.Height, 100, 1000);
            float[,] ar = result.Make2D(bmp.Height, bmp.Width);
            var bmp2 = ImageHelper.SaveArrayToBitmap(ar);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}

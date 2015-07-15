using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUDAFingerprinting.GPU.Tests
{
    [TestClass]
    public class ImageEnhancementTest
    {
        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Enhance")]
        public static extern void Enhance(float[,] source, int imgWidth, int imgHeight, float[] res, float[,] orientationMatrix,
        float frequency, int filterSize, int angleNum);

        [DllImport("CUDAFingerprinting.GPU.OrientationField.dll", CallingConvention = CallingConvention.Cdecl,
            EntryPoint = "OrientationFieldInPixels")]
        public static extern void OrientationFieldInPixels(float[] res, float[,] floatArray, int width, int height);
        [TestMethod]
        public void EnhanceTest()
        {
            var bmp = Resources.SampleFinger2;
            float[,] array = ImageHelper.LoadImageToFloats(bmp);
            
            float[] orientLin = new float[bmp.Width * bmp.Height];
            OrientationFieldInPixels(orientLin, array, array.GetLength(1), array.GetLength(0));
            float[,] orient = orientLin.Make2D(bmp.Height, bmp.Width);

            float[] result = new float[bmp.Width * bmp.Height];
            Enhance(array, array.GetLength(1), array.GetLength(0), result, orient, (float) 1 / 9, 32, 8);

            float[,] ar = result.Make2D(bmp.Height, bmp.Width);
            var bmp2 = ImageHelper.SaveArrayToBitmap(ar);
            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}
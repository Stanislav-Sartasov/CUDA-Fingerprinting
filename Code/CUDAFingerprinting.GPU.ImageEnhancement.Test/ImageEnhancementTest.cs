using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;
namespace CUDAFingerprinting.GPU.Tests
{
    [TestClass]
    public class ImageEnhancementTest
    {
        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Enhance32")]
        public static extern void Enhance(float[,] source, int imgWidth, int imgHeight, float[] res, float[,] orientationMatrix,
        float frequency, int filterSize, int angleNum);

        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Enhance32")]
        public static extern void Enhance32(float[,] source, int imgWidth, int imgHeight, float[] res, float[,] orientationMatrix,
        float frequency, int angleNum);

        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Enhance16")]
        public static extern void Enhance16(float[,] source, int imgWidth, int imgHeight, float[] res, float[,] orientationMatrix,
        float frequency, int angleNum);

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
            Enhance(array, array.GetLength(1), array.GetLength(0), result, orient, (float)1 / 9, 22, 8);

            float[,] ar = result.Make2D(bmp.Height, bmp.Width);
            var bmp1 = ImageHelper.SaveArrayToBitmap(ar);
            bmp1.Save("testUnder32Filter.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));

            float[] result2 = new float[bmp.Width * bmp.Height];
            Enhance16(array, array.GetLength(1), array.GetLength(0), result2, orient, (float)1 / 9, 8);

            float[,] ar2 = result2.Make2D(bmp.Height, bmp.Width);
            var bmp2 = ImageHelper.SaveArrayToBitmap(ar2);
            bmp2.Save("test16x16Filter.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));

            float[] result3 = new float[bmp.Width * bmp.Height];
            Enhance32(array, array.GetLength(1), array.GetLength(0), result3, orient, (float)1 / 9, 8);

            float[,] ar3 = result3.Make2D(bmp.Height, bmp.Width);
            var bmp3 = ImageHelper.SaveArrayToBitmap(ar3);
            bmp3.Save("test32x32Filter.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}

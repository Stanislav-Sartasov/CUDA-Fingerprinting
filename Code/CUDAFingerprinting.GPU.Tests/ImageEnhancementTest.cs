using System;
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
            EntryPoint = "OrientatiobFieldInPixels")]
        public static extern void OrientatiobFieldInPixels(float[] res, float[,] floatArray, int width, int height);
        [TestMethod]
        public void CreateGaborFilter16Test()
        {
            var bmp = Resources.SampleFinger1;
            float[,] array = ImageHelper.LoadImageToFloats(bmp);
            float[] result = new float[bmp.Width * bmp.Height];
            float[] orientLin = new float[bmp.Width * bmp.Height];
            OrientatiobFieldInPixels(orientLin, array, array.GetLength(0), array.GetLength(1));
            float[,] orient = orientLin.Make2D(bmp.Height, bmp.Width);
            Enhance(array, array.GetLength(0), array.GetLength(1), result, orient, (float) 1 / 9, 16, 8);
            float[,] ar = result.Make2D(bmp.Height, bmp.Width);
            var bmp2 = ImageHelper.SaveArrayToBitmap(ar);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}

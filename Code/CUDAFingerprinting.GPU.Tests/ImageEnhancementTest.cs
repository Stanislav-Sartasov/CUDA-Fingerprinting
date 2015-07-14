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
        public static extern void Enhance(float[,] source, int imgWidth, int imgHeight, float[] res, float[] orientationMatrix,
        float frequency, int filterSize, int angleNum);

        [TestMethod]
        public void CreateGaborFilter16Test()
        {
            var bmp = Resources.SampleFinger1;
            float[,] array = ImageHelper.LoadImageToFloats(bmp);
            float[] result = new float[bmp.Width * bmp.Height];

            ImageHelper.SaveArray(filters.Make2D(16, 16 * 8), "test.bmp", true);
        }
    }
}

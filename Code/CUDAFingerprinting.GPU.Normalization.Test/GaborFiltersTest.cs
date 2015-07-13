using System;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.GPU.Tests
{
    [TestClass]
    public class GaborFiltersTest
    {
        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "MakeGaborFilters")]
        public static extern void MakeGaborFilters(float[] filter, int size, int angleNum, float frequency);

        [TestMethod]
        public void CreateGaborFilter0Test()
        {
            var filter0 = new float[16 * 16];
            MakeGaborFilters(filter0, 16, 8, (float) 1 / 9);

            ImageHelper.SaveArray(filter0.Make2D(16, 16), "F0.bmp", true);
        }
    }
}

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
            var filters = new float[16 * 16 * 8];

            MakeGaborFilters(filters, 16, 8, (float) 1 / 9);

            ImageHelper.SaveArray(filters.Make2D(16 * 8, 16), "test.bmp", true);
        }
    }
}

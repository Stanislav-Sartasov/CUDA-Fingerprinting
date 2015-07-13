using System;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace CUDAFingerprinting.GPU.Tests
{
    [TestClass]
    public class GaborFiltersTest
    {
        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "MakeGabor16Filters")]
        public static extern void MakeGabor16Filters(float[] filter, int angleNum, float frequency);
        [DllImport("CUDAFingerprinting.GPU.Filters.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "MakeGabor32Filters")]
        public static extern void MakeGabor32Filters(float[] filter, int angleNum, float frequency);

        [TestMethod]
        public void CreateGaborFilter16Test()
        {
            var filters = new float[16 * 16 * 8];

            MakeGabor16Filters(filters, 8, (float) 1 / 9);

            ImageHelper.SaveArray(filters.Make2D(16 * 8, 16), "test.bmp", true);
        }
    }
}

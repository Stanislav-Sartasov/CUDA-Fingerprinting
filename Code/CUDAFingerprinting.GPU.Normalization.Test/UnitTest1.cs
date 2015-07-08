using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Text;
using System.Drawing;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;


namespace CUDAFingerprinting.GPU.Normalization.Test
{
    [TestClass]
    public class UnitTest1
    {
        [DllImport("CUDAFingerprinting.GPU.Normalization.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Normalize")]
       // [return: MarshalAs(UnmanagedType.SafeArray)]
        public static extern void Normalize(float[] source, float[] res, int imgWidth, int imgHeight, int bordMean, int bordVar);
        //static void WriteFloat(IntPtr ptr, float value)
        //{
        //    foreach (var b in BitConverter.GetBytes(value))
        //    {
        //        Marshal.WriteByte(ptr, b);
        //        ptr += 1;
        //    }
        //}

        [TestMethod]
        public void NormalizationTest()
        {
            var bmp = Resources.SampleFinger1;
            float[,] array0 = ImageHelper.LoadImageToFloats(bmp);
            float[] array = array0.Make1D();
            float[] result = new float[bmp.Width * bmp.Height];
            Normalize(array, result, bmp.Width, bmp.Height, 100, 1000);
            //IntPtr ptr = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);
            //float[] result = new float[bmp.Width * bmp.Height];
            //float a = (float)1.0;
            //WriteFloat(ptr, a);
            //Marshal.Copy(ptr, result, 0, bmp.Width * bmp.Height);
            //array = Normalize(array, bmp.Width, bmp.Height, 1000, 1000);
            float[,] ar = result.Make2D(bmp.Height, bmp.Width);
            var bmp2 = ImageHelper.SaveArrayToBitmap(ar);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}

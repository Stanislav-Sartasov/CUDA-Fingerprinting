using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.IO;
using System.Text;
using System.Drawing;
using System.Runtime.InteropServices;

namespace CUDA
{
    public class NormalizeClass
    {
        [DllImport("CUDAFingerprinting.GPU.Normalisation.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "Normalize")]
        public static extern float[,] Normalize(float[,] source, int imgWidth, int imgHeight, int bordMean, int bordVar);
    }
}

namespace CUDAFingerprinting.GPU.Normalization.Test
{
    [TestClass]
    public class UnitTest1
    {
        
        [TestMethod]
        public void NormalizationTest()
        {
            var bmp = Resources.SampleFinger2;
            var array = ImageLoading.LoadBmp(bmp);

            array = array.DoNormalization(1000, 1000);

            var bmp2 = ImageHelper.SaveArrayToBitmap(array);

            bmp2.Save("test.bmp", ImageHelper.GetImageFormatFromExtension("test.bmp"));
        }
    }
}

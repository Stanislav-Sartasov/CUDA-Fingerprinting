using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System.Runtime.InteropServices;
using CUDAFingerprinting.Common;
using System.IO;

namespace CUDAFingerprinting.GPU.ImageBinarization.CUDATest
{
    [TestClass]
    public class ImageBinarizationTest
    {
        [DllImport("CUDAFingerprinting.GPU.ImageBinarization.dll", CallingConvention = CallingConvention.Cdecl, EntryPoint = "BinarizeImage")]
        public static extern void BinarizeImage(int line, int[,] src, int[,] dist, int width, int height);
        [TestMethod]
        public void BinarizationTest()
        {
            int[,] image = ImageHelper.LoadImage<int>(Resources._2_6);
            int width = image.GetLength(0);
            int height = image.GetLength(1);
            int[,] dist = new int[width, height];
            BinarizeImage(128, image, dist, width, height);
            ImageHelper.SaveArrayToBitmap(dist).Save(Path.GetTempPath() + Guid.NewGuid() + ".bmp");
        }
    }
}
